import json
import logging
import os
import argparse
import pickle
from typing import Dict
import random

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import InstructBlipProcessor, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.dataset import load_datasets, CustomDataCollator, collator, Recipe1M_Collator, load_datasets_for_distributed
from data.utils import Vocabulary, to_one_hot
from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration
from common.dist_utils import init_distributed_mode
from common.logger import setup_logger

from transformers.trainer_utils import EvalPrediction

# TODO
# 2. compute_metric : F1/IoU
# 5. log only main process

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='vicuna')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-vicuna-7b',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl', "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'(default)."
    )

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Call the parent method
        batch = super().__call__(features)

        # Ensure ingredient_ids are included in the batch
        ingredient_ids = [feature['ingredient_ids'] for feature in features]
        batch['ingredient_ids'] = ingredient_ids

        return batch

class DecoderEvalMetrics():
    
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.ingr2id = pickle.load(open('/nfs_share2/shared/from_donghee/recipe1m_data/recipe1m_vocab_ingrs.pkl', 'rb')).word2idx
        label_ids2ingr_class = dict()
        for entry in eval_dataset:
            label_ids2ingr_class[tuple(entry['label_ids'])] = entry['ingredient_int']
        self.label_ids2ingr_class = label_ids2ingr_class

    def map_to_classes(self, batch_tokens, max_len=20):
        ingredient_text = self.tokenizer.batch_decode(batch_tokens)
        
        # Process all ingredients in a batch together
        batch_ingr_ids = []
        for ingrs in ingredient_text:
            ingr_text = [ingr.strip().replace(' ', '_') for ingr in ingrs.split(',')]
            ingr_ids = [self.ingr2id.get(ingr, None) for ingr in ingr_text if ingr in self.ingr2id]
            # batch_ingr_ids.append(ingr_ids)

            # Pad the list to ensure consistent length
            if max_len > len(ingr_ids):
                padded_ingr_ids = ingr_ids + [self.ingr2id.get("<pad>", -1)] * (max_len - len(ingr_ids))
            else:
                padded_ingr_ids = ingr_ids
            batch_ingr_ids.append(padded_ingr_ids[:max_len])  # Ensures the list is not longer than max_len

        return batch_ingr_ids

    def compute_metrics(self, pred, tokenized_pred=False, verbose=True):
        labels = pred.label_ids # text_output ids
        target_ingr = []
        for label in labels:
            ingrs = self.label_ids2ingr_class[tuple(label)]
            target_ingr.append(ingrs)
        
        target_ingr = torch.tensor(target_ingr) # one-hot already
        target_ingr = to_one_hot(target_ingr)

        if tokenized_pred:
            pred_ingr = self.map_to_classes(pred.predictions)
        else:
            pred_ingr = self.map_to_classes(pred.predictions[0].argmax(-1))
        pred_ingr = to_one_hot(torch.tensor(pred_ingr))
        
        f1_micro = f1_score(target_ingr, pred_ingr, average='micro')
        f1_macro = f1_score(target_ingr, pred_ingr, average='macro')
        iou_micro = jaccard_score(target_ingr, pred_ingr, average='micro')
        iou_macro = jaccard_score(target_ingr, pred_ingr, average='macro')
        # acc = accuracy_score(target_ingr, pred_ingr)

        result = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            # 'accuracy': acc,
            'iou_macro': iou_macro,
            'iou_micro': iou_micro,
        }

        if verbose:
            logging.info(f'* Evaluation result: {result}')

        return result

class MyTrainer(Trainer):
    def __init__(
        self,
        model= None,
        args= None,
        data_collator= None,
        train_dataset= None,
        eval_dataset= None,
        tokenizer= None,
        model_init= None,
        compute_metrics = None,
        callbacks= None,
        optimizers= (None, None),
        preprocess_logits_for_metrics = None,
        metric_class=None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.metric_class = metric_class
        
    
    # TODO optimize for multiprocess
    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys= None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        
        with torch.no_grad():
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
                )
            
            self.model.eval()

            f1s = []
            ious = []
            
            for start_idx in range(0, len(self.eval_dataset), self._train_batch_size):
                end_idx = min(len(self.eval_dataset), start_idx + self._train_batch_size)
                
                batch = {
                    'pixel_values': torch.tensor(self.eval_dataset['pixel_values'][start_idx:end_idx]).to(self.model.device),
                    'input_ids': torch.tensor(self.eval_dataset['input_ids'][start_idx:end_idx]).to(self.model.device),
                    'attention_mask': torch.tensor(self.eval_dataset['attention_mask'][start_idx:end_idx]).to(self.model.device),
                    'qformer_input_ids': torch.tensor(self.eval_dataset['qformer_input_ids'][start_idx:end_idx]).to(self.model.device),
                    'qformer_attention_mask': torch.tensor(self.eval_dataset['qformer_attention_mask'][start_idx:end_idx]).to(self.model.device),
                }
                
                outputs = self.model.generate(
                        **batch,
                        do_sample = False,
                        num_beams=5,
                        max_length=128,
                        min_length=1,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                    )
                
                gen_metrics = self.metric_class.compute_metrics(
                    EvalPrediction(predictions=outputs.cpu(), label_ids=self.eval_dataset['label_ids'][start_idx:end_idx]),
                    tokenized_pred=True,
                    verbose=False,
                )
                f1s.append(gen_metrics['f1_micro'])
                ious.append(gen_metrics['iou_micro'])

                if random.random() < 0.3:
                    rand_idx = random.randint(0, len(outputs)-1)
                    gen_text = self.metric_class.tokenizer.decode(outputs[rand_idx].cpu())
                    logging.info(f'- Generation example: ,{gen_text}')

                del batch, outputs
        
        torch.cuda.empty_cache()
        
        gen_metrics = {
            'gen_f1': sum(f1s) / len(f1s),
            'gen_iou': sum(ious) / len(ious)
        }
        metrics.update(gen_metrics)
        
        logging.info('==================================')
        logging.info(f'* Evaluation result: {metrics}')
        
        return metrics


def train(args):
    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    datasets = load_datasets( 
        processor=processor, 
        data_dir=args.dataset_path, 
        training_samples=args.training_samples,
        eval_samples=args.eval_samples, 
        pre_map=args.pre_map,
        decoder_only=args.decoder_only
    )
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        do_train=False,
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        # metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        # remove_unused_columns= False ## TODO
    )

    eval_metrics = DecoderEvalMetrics(processor.tokenizer, datasets['val'])
    # TODO: compute_metrics
    trainer = MyTrainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        # data_collator = CustomDataCollator(tokenizer=processor.tokenizer, model=model)
        compute_metrics=eval_metrics.compute_metrics,
        # data_collator=data_collator
        metric_class = eval_metrics,
    )

    # Train the model
    trainer.train()

    # eval_result = trainer.evaluate(datasets['val'])
    # print("EVAL")
    # print(eval_result)

    # Save
    # processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    # test_results = trainer.evaluate(datasets['test'])
    # print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ###
    args.batch_size = 4
    args.training_samples = 100
    args.eval_samples = 10
    args.eval_steps = 10
    args.logging_steps = 50
    ###

    pretty_print(args)

    train(args)
