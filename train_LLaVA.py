import json
import logging
import os
import argparse
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

import torch
# from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import TrainingArguments, Trainer, AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.dataset import llava_load_datasets
from data.utils import Vocabulary
from model.modeling_LLaVA import FreezeLlavaForConditionalGeneration
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre

# TODO
# 5. log only main process /

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for LLaVA.")

    parser.add_argument('--project_name', type=str, default='temp') ## TODO 'clip_BERT'
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--eval_steps', type=int, default=500) 
    parser.add_argument('--logging_steps', type=int, default=50) 
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--test_samples', type=int, default=1500, help='number of test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--load_from_cache_file', type=bool, default=True, help='load dataset from huggingface cache')
    parser.add_argument('--train_llm', type=bool, default=False, help='train llm backbone')
    parser.add_argument('--train_vit', type=bool, default=False, help='train ViT')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='llava-hf/llava-1.5-7b-hf',
        choices=['llava-hf/llava-1.5-7b-hf', 'liuhaotian/llava-v1.5-7b', 'liuhaotian/llava-v1.5-13b', 'liuhaotian/llava-v1.5-7b-lora', 'liuhaotian/llava-v1.5-13b-lora'],
    )
    parser.add_argument(
        '--bert_name', 
        type=str, 
        default='bert-large-uncased',
        choices=['bert-large-uncased', 'bert-base-uncased'],
        help="Specifies the BERT model to use. Choose from 'bert-large-uncased' (default), "
            "or 'bert-base-uncased'."
    )
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14-336',help="clip-vit model")

    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()

    if args.bert_name is not None:
        args.encoder_only = True
    
    args.output_dir= os.path.join("./outputs", args.project_name) ## TODO if exist, add calendar time
    args.logging_dir = os.path.join('./logs', args.project_name)

    # if 't5' in args.model_name:
    #     args.decoder_only = False
    # else:
    #     args.decoder_only = True
    args.decoder_only = True ## TODO not hard code

    return args

def compute_metrics(pred): # TODO location
    labels = pred.label_ids
    preds = pred.predictions

    if len(preds.shape) == 2: # 2D
        preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy() >= 0.5
    else: # 3D
        preds = torch.sigmoid(torch.tensor(pred.predictions[0])).numpy() >= 0.5

    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    iou_macro = jaccard_score(labels, preds, average='macro') # TODO macro iou?
    iou_micro = jaccard_score(labels, preds, average='micro')

    result = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': acc,
        'iou_macro': iou_macro,
        'iou_micro': iou_micro,
    }

    logging.info(f'* Evaluation result: {result}')

    return result

def train(args):
    model = FreezeLlavaForConditionalGeneration.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    # TODO input ids, labels 어떤 식으로 config 해야하는지..
    datasets = llava_load_datasets( 
        processor=processor, 
        data_dir=args.dataset_path, 
        training_samples=args.training_samples,
        eval_samples=args.eval_samples, 
        test_samples=args.test_samples,
        pre_map=args.pre_map,
        decoder_only=args.decoder_only,
        encoder_only = args.encoder_only,
        load_from_cache_file = args.load_from_cache_file
    )
    
    processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps = args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        # metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        # compute_metrics=compute_metrics_thre # TODO compute_metrics - f1, iou
        # data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)
        # callbacks=[PrinterCallback]
    )

    # Train the model
    # TODO forward 에서 input_ids 에 32000 (self.config.image_token_id?) 이거 없는데.. 자꾸 그걸로 이미지 개수를 센대..
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # eval_result = trainer.evaluate()
    # print("EVAL")
    # print(eval_result)


    # Save
    
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    # with open('BERT_f1_threshold.json', 'w') as f:
    #     json.dump(test_results, f, indent=4)
    print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ####
    args.training_samples = 64
    args.epochs = 10
    # args.train_llm = False
    # args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/clip_BERT/checkpoint-17500'
    args.batch_size = 4
    # args.train_vit = True
    # args.eval_steps = 10
    args.eval_samples = 100
    args.test_samples = 100
    ####

    pretty_print(args)

    train(args)
