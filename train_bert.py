import json
from transformers.utils import logging
import os
import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

from data.dataset import load_datasets, load_from_cache
from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from model.modeling_instructblip import BERTInstructBlipForConditionalGeneration
from model.processing_instructblip import BERTInstructBlipProcessor
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre, compute_metrics_acc, compute_metrics_fix_thre
from datasets import load_dataset, DatasetDict
from common.snapme import test_snapme
from common.utils import load_eval_checkpoint

# TODO

logger = logging.get_logger(__name__)

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='BERT_final') # 'BERT
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')
    parser.add_argument('--dataset_name', type=str, default='recipe1m', choices=['recipe1m', 'mnist', 'cifar10', 'cifar100', 'cifar100_fine_label'], help='Hugging face built-in datasets or Recipe1M')
    parser.add_argument('--dataset_cache_path', type=str, default='/home/donghee/huggingface_data_cache', help='local dataset cache directory')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--eval_steps', type=int, default=500) 
    parser.add_argument('--logging_steps', type=int, default=50) 
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--test_samples', type=int, default=-1, help='number of test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--load_from_cache_file', type=bool, default=True, help='load dataset from huggingface cache')
    parser.add_argument('--train_llm', type=bool, default=False, help='train llm backbone')
    parser.add_argument('--train_vit', type=bool, default=False, help='train ViT')
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
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()

    if args.bert_name is not None:
        args.encoder_only = True
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args


def train(args):
    
    processor = BERTInstructBlipProcessor.from_pretrained(args.model_name) # TODO - better way
    processor.to_bert(args.bert_name) 

    multi_classification = True if args.dataset_name == 'recipe1m' else False ##
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
    datasets, num_labels = load_from_cache(args, processor, possible_cache_dir)

    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    model = BERTInstructBlipForConditionalGeneration(args.bert_name, args.train_llm, args.train_vit)

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
        compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        # data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    logger.info("* Test start *")
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    logger.info(test_results)


def eval(args):
    processor = BERTInstructBlipProcessor.from_pretrained(args.model_name) # TODO - better way
    processor.to_bert(args.bert_name) 

    multi_classification = True if args.dataset_name == 'recipe1m' else False ##
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
    datasets, num_labels = load_from_cache(args, processor, possible_cache_dir)

    model = BERTInstructBlipForConditionalGeneration(args.bert_name, args.train_llm, args.train_vit)

    logger.info("** Eval only mode **")
    model = load_eval_checkpoint(model, args.eval_checkpoint)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
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
        compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        # data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)
    )

    if multi_classification: # TODO save best_threshold during training
        # decide threshold
        logger.info("* Eval start (to decide threshold) *")
        eval_results = trainer.evaluate(datasets['val'])
        best_threshold = eval_results['eval_max_threshold']
        logger.info(f"* Best threshold: {best_threshold}") # 0.25
        trainer.compute_metrics = lambda pred: compute_metrics_fix_thre(pred, threshold=best_threshold)

    # logger.info("* Test start *")
    # datasets['test'] = datasets['test'].select(range(200))
    # test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    # logger.info(test_results)

    if args.snapme_test:
        logger.info("* SNAPMe eval *")
        snapme_result, beverage_result, non_beverage_result = test_snapme(processor, trainer, args.dataset_cache_path)
        logger.info(snapme_result)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ####
    # args.training_samples = 64
    args.project_name = 'temp'
    args.epochs = 20
    args.train_llm = False
    # args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/BERT_train2/checkpoint-5500'
    args.batch_size = 64
    # args.train_vit = True
    # args.eval_steps = 10
    # args.eval_samples = 100
    # args.test_samples = 10000
    args.eval_only = True
    args.eval_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/BERT_final/best'
    ####

    pretty_print(args)

    if args.eval_only:
        eval(args)
    else:
        train(args)
