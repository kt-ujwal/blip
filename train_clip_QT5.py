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
from transformers import InstructBlipProcessor, TrainingArguments, Trainer, DataCollatorForSeq2Seq, InstructBlipConfig
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.dataset import load_datasets, CustomDataCollator, collator, Recipe1M_Collator, load_datasets_for_distributed
from data.utils import Vocabulary, to_one_hot
from model.processing_instructblip import CLIP_QT5InstructBlipProcessor
from model.modeling_instructblip import CLIP_QT5InstructBlipForConditionalGeneration
from common.dist_utils import init_distributed_mode
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre

from transformers.trainer_utils import EvalPrediction

# TODO
# 5. log only main process

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='clip_QT5')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--num_query', type=int, default=8, help='number of learnable query passed to decoder')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-large-patch14-336',help="clip-vit model")

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args


def train(args):
    config = InstructBlipConfig()
    model = CLIP_QT5InstructBlipForConditionalGeneration(config)

    # model = CLIP_QT5InstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    # model.reinit(num_query=args.num_query, clip_model=args.clip_model)

    processor = CLIP_QT5InstructBlipProcessor.from_pretrained(args.model_name)
    processor.to_clip(args.clip_model) # TODO better way
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
        do_train=True, ## True !!
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
    # print("No TRAIN!!")
    # TODO: compute_metrics
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        compute_metrics=compute_metrics_thre, ## compute_metrics
    )

    # Train the model
    trainer.train()

    eval_result = trainer.evaluate(datasets['val'])
    print("EVAL")
    print(eval_result)

    # Save
    # processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ###
    args.batch_size = 64
    # args.training_samples = 64
    # args.eval_samples = 64
    # args.eval_steps = 5
    # args.logging_steps = 50
    args.epochs = 20
    args.num_query = 1
    args.project_name = 'clip_QT5_1'
    args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/clip_QT5/checkpoint-4000'
    ###

    pretty_print(args)

    train(args)
