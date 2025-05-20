import json
from transformers.utils import logging
import os
import argparse

from transformers import InstructBlipProcessor, BERTInstructBlipProcessor, TrainingArguments, Trainer, EarlyStoppingCallback

from data.dataset import load_from_cache
from data.utils import Vocabulary
from model.modeling_instructblip import QformerInstructBlip, FreezeInstructBlipForConditionalGeneration, BERTInstructBlipForConditionalGeneration

from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre, compute_metrics_acc, compute_metrics_fix_thre, Recipe1mEvalMetrics
from common.snapme import test_snapme
from common.utils import load_eval_checkpoint
from common.trainer import CustomTrainer

logger = logging.get_logger(__name__)

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training script for instructBlip BERT, Qformer, and T5 models.")
    
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['BERT', 'Qformer', 'T5'], help='Type of model to train: BERT, Qformer, or T5.')
    parser.add_argument('--dataset_path', type=str, required=True, help='path containing Recipe1M dataset')

    parser.add_argument('--dataset_name', type=str, default='recipe1m', choices=['recipe1m', 'mnist', 'cifar10', 'cifar100', 'cifar100_fine_label'], help='Hugging face built-in datasets or Recipe1M')
    parser.add_argument('--dataset_cache_path', type=str, default='./huggingface_data_cache', help='local dataset cache directory')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--train_llm', type=bool, default=False, help='train llm backbone')
    parser.add_argument('--train_vit', type=bool, default=False, help='train ViT')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32) # 16
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--num_labels', type=int, default=1488, help='number of labels for classification')
    parser.add_argument('--freeze_qformer', type=bool, default=False, help='if True, qformer is being freeze during training')
    parser.add_argument('--fine_label', type=bool, default=True, help='if True, use fine labels for classification')
    parser.add_argument('--eval_split_ratio', type=float, default=0.1, help='split ratio for validation set')
    parser.add_argument('--generate_mode', type=bool, default=False, help='True for generation task, False for classification task')
    parser.add_argument('--eval_only', type=bool, default=False, help='if True, only eval on Test set. no train')
    parser.add_argument('--eval_checkpoint', type=str, default=None, help='path contains pytorch_model.bin files')
    parser.add_argument('--snapme_test', type=bool, default=True)

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )
    parser.add_argument(
        '--bert_name', 
        type=str, 
        default='bert-large-uncased',
        choices=['bert-large-uncased', 'bert-base-uncased'],
        help="Specifies the BERT model to use. Choose from 'bert-large-uncased' (default), "
            "or 'bert-base-uncased'."
    )

    args = parser.parse_args()
    
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args, model, processor, datasets, multi_classification):
    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size else args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model, # TODO multi-label classification metrics.. mAP? AP? loss?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        remove_unused_columns=args.remove_unused_columns,
        # include_inputs_for_metrics=True,
    )

    if args.model_type == 'T5':
        eval_metrics = Recipe1mEvalMetrics(processor.tokenizer)
    
        trainer = CustomTrainer( 
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            tokenizer=processor.tokenizer,
            compute_metrics=eval_metrics.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

    else:
        trainer = Trainer( 
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    logger.info("* Test start *")
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    logger.info(test_results)

    if args.snapme_test: 
        logger.info("* SNAPMe eval *")
        snapme_result = test_snapme(processor, trainer, args.dataset_cache_path)
        logger.info(snapme_result)
    
    logger.info("** Training Done **")

def eval(args, model, processor, datasets, multi_classification):
    logger.info("** Eval only mode **")
    model = load_eval_checkpoint(model, args.eval_checkpoint)

    training_args = TrainingArguments( # TODO not necessary. simplify
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size else args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model, # TODO multi-label classification metrics.. mAP? AP?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        remove_unused_columns=args.remove_unused_columns,
        # include_inputs_for_metrics=True,
    )

    if args.model_type == 'T5':
        eval_metrics = Recipe1mEvalMetrics(processor.tokenizer) # TODO turn to lambda function
    
        trainer = CustomTrainer( 
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            tokenizer=processor.tokenizer,
            compute_metrics=eval_metrics.compute_metrics,
        )

    else:
        trainer = Trainer( 
            model=model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['val'],
            compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc,
        )

    if multi_classification: # TODO save best_threshold during training
        # decide threshold
        logger.info("* Eval start (to decide threshold) *")
        eval_results = trainer.evaluate(datasets['val'])
        best_threshold = eval_results['eval_max_threshold']
        logger.info(f"* Best threshold: {best_threshold}")
        trainer.compute_metrics = lambda pred: compute_metrics_fix_thre(pred, threshold=best_threshold)

    logger.info("* Test start *")
    # datasets['test'] = datasets['test'].select(range(200))
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    logger.info(test_results)

    if args.snapme_test:
        logger.info("* SNAPMe eval *")
        snapme_result, beverage_result, non_beverage_result = test_snapme(processor, trainer, args.dataset_cache_path)
        logger.info(snapme_result)
    
    logger.info("** Eval Done **")

def main():
    args = parse_args()
    
    setup_logger(args)
    pretty_print(args)

    if args.model_type == 'T5':
        args.generate_mode = True

        processor = InstructBlipProcessor.from_pretrained(args.model_name)
        model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)

        possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
        datasets, num_labels = load_from_cache(args, processor, possible_cache_dir)
        
        args.metric_for_best_model = 'f1_micro'
        args.remove_unused_columns = False
        # multi_classification = None

    elif args.model_type == 'BERT':
        processor = BERTInstructBlipProcessor.from_pretrained(args.model_name)
        processor.to_bert(args.bert_name) 
        model = BERTInstructBlipForConditionalGeneration(args.bert_name, args.train_llm, args.train_vit, num_labels)

        possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
        datasets, num_labels = load_from_cache(args, processor, possible_cache_dir)

        multi_classification = True if args.dataset_name == 'recipe1m' else False
        args.metric_for_best_model = 'loss' if multi_classification else 'accuracy' # TODO max_f1
        args.remove_unused_columns = True
        # args.compute_metrics = compute_metrics_thre if multi_classification else compute_metrics_acc

    else: # Qformer
        processor = InstructBlipProcessor.from_pretrained(args.model_name)
        model = QformerInstructBlip.from_pretrained(args.model_name) 
        model.remove_llm(num_labels=num_labels, multi_classification=multi_classification)

        possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
        datasets, num_labels = load_from_cache(args, processor, possible_cache_dir, remove_columns=['input_ids', 'attention_mask'])

        multi_classification = True if args.dataset_name == 'recipe1m' else False
        args.metric_for_best_model = 'max_f1' if multi_classification else 'accuracy'
        args.remove_unused_columns = True
        # args.compute_metrics = compute_metrics_thre if multi_classification else compute_metrics_acc

    if args.eval_only:
        eval(args, model, processor, datasets, multi_classification)
    else:
        train(args, model, processor, datasets, multi_classification)
    
    logger.info("** DONE **")

if __name__ == '__main__':
    main()