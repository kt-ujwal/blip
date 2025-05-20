import json
# import logging
from PIL import Image   
from transformers.utils import logging
import os
import argparse

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import InstructBlipProcessor, TrainingArguments, Trainer, DataCollatorForSeq2Seq, InstructBlipConfig, InstructBlipForConditionalGeneration, EarlyStoppingCallback

# from torchvision.datasets import VOCDetection
from data.dataset import load_datasets, load_from_cache
from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from model.modeling_instructblip import QformerInstructBlip
from common.dist_utils import init_distributed_mode
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre, compute_metrics_acc, compute_metrics_fix_thre

from transformers import T5ForSequenceClassification
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset, DatasetDict, Dataset
from data.utils import to_one_hot, get_cache_file_name
from common.snapme import test_snapme
from common.utils import load_eval_checkpoint

# TODO
# 1. general datasets

logger = logging.get_logger(__name__)

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='temp')
    # /path/to/Recipe1M/dataset
    # /nfs_share2/shared/from_donghee/recipe1m_data
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')
    parser.add_argument('--dataset_name', type=str, default='recipe1m', choices=['recipe1m', 'mnist', 'cifar10', 'cifar100', 'cifar100_fine_label'], help='Hugging face built-in datasets or Recipe1M')
    parser.add_argument('--dataset_cache_path', type=str, default='/home/donghee/huggingface_data_cache', help='local dataset cache directory')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--num_query', type=int, default=8, help='number of learnable query passed to decoder')
    # parser.add_argument('--num_labels', type=int, default=1488, help='number of labels for classification')
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

    args = parser.parse_args()
    
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args):
    # TODO better way to reinit
    
    # model = InstructBlipForConditionalGeneration.from_pretrained(args.model_name)

    # for m in model.modules():
    #     for param in m.parameters():
    #         param.requires_grad = False
    
    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    
    # TODO idenity multi-label classification
    multi_classification = True if args.dataset_name == 'recipe1m' else False
    ##
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)

    datasets, num_labels = load_from_cache(args, processor, possible_cache_dir, remove_columns=['input_ids', 'attention_mask'])

    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    model = QformerInstructBlip.from_pretrained(args.model_name) ## TODO load from disk
    model.remove_llm(num_labels=num_labels, multi_classification=multi_classification)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size else args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='max_f1' if multi_classification else 'accuracy', # TODO multi-label classification metrics.. mAP? AP?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        # remove_unused_columns= False ## TODO
    )
    
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
    
    logger.info("* Eval start *")
    eval_result = trainer.evaluate(datasets['val'])
    logger.info(eval_result)

    logger.info("* Test start *")
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    logger.info(test_results) # TODO save test_result

    if args.snapme_test: # TODO debug
        logger.info("* SNAPMe eval *")
        snapme_result = test_snapme(processor, trainer, args.dataset_cache_path)
        logger.info(snapme_result)

def eval(args):
    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    
    multi_classification = True if args.dataset_name == 'recipe1m' else False
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
    datasets, num_labels = load_from_cache(args, processor, possible_cache_dir, remove_columns=['input_ids', 'attention_mask'])

    model = QformerInstructBlip.from_pretrained(args.model_name) 
    model.remove_llm(num_labels=num_labels, multi_classification=multi_classification)

    logger.info("** Eval only mode **")
    model = load_eval_checkpoint(model, args.eval_checkpoint)

    training_args = TrainingArguments( # TODO not necessary
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size else args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='max_f1' if multi_classification else 'accuracy', # TODO multi-label classification metrics.. mAP? AP?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        # remove_unused_columns= False ## TODO
    )
    
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    if multi_classification: # TODO save best_threshold during training
        # decide threshold
        logger.info("* Eval start (to decide threshold) *")
        eval_results = trainer.evaluate(datasets['val'])
        best_threshold = eval_results['eval_max_threshold']
        logger.info(f"* Best threshold: {best_threshold}") # 0.3
        trainer.compute_metrics = lambda pred: compute_metrics_fix_thre(pred, threshold=best_threshold)


    logger.info("* Test start *")
    # datasets['test'] = datasets['test'].select(range(200))
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    logger.info(test_results) # 0.571

    if args.snapme_test: # TODO debug
        logger.info("* SNAPMe eval *")
        snapme_result, beverage_result, non_beverage_result = test_snapme(processor, trainer, args.dataset_cache_path)
        logger.info(snapme_result)



if __name__ == '__main__':
    args = parse_args()

    ###
    args.project_name = 'temp'
    args.batch_size = 128 # batchsize 32, num_query 4 : 33GB, batchsize 16 num_query 1: 24GB
    args.eval_only = True
    args.eval_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/train_Qformer_only_recipe1m/best'
    args.snapme_test = True
    # args.eval_batch_size = 128
    # args.training_samples = 32
    # args.eval_samples = 128
    # args.eval_steps = 5
    # args.logging_steps = 5
    # args.epochs = 1
    # args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/t5_learnable_query8/checkpoint-20500'
    ###

    setup_logger(args)
    pretty_print(args)

    if args.eval_only:
        eval(args)
    else:
        train(args)
