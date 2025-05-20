import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import json
# import logging
import os
import argparse

import torch
from transformers import InstructBlipProcessor, TrainingArguments, EarlyStoppingCallback
from transformers.utils import logging 

from data.dataset import load_datasets, load_from_cache
from datasets import load_dataset, DatasetDict
from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration

from common.logger import setup_logger
from common.compute_metrics import Recipe1mEvalMetrics
from common.trainer import CustomTrainer
from common.snapme import test_snapme
from common.utils import load_eval_checkpoint

# TODO
# 5. log only main process (logging, logger)

logger = logging.get_logger(__name__)

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='T5_f1')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')
    parser.add_argument('--dataset_name', type=str, default='recipe1m', choices=['recipe1m', 'mnist', 'cifar10', 'cifar100'], help='Hugging face built-in datasets or Recipe1M')
    parser.add_argument('--dataset_cache_path', type=str, default='/home/donghee/huggingface_data_cache', help='local dataset cache directory')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--fine_label', type=bool, default=False, help='if True, use fine labels for classification')
    parser.add_argument('--eval_split_ratio', type=float, default=0.1, help='split ratio for validation set')
    parser.add_argument('--generate_mode', type=bool, default=True, help='True for generation task, False for classification task')
    parser.add_argument('--eval_only', type=bool, default=False, help='if True, only eval on Test set. no train')
    parser.add_argument('--eval_checkpoint', type=str, default=None, help='path contains pytorch_model.bin files')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
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
    
    # args.output_dir= os.path.join("./outputs", args.project_name)
    # args.logging_dir = os.path.join('./logs', args.project_name)

    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args):
    """
    Training script for original InstructBlip with T5-xl
    """

    processor = InstructBlipProcessor.from_pretrained(args.model_name)

    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
    
    datasets, _ = load_from_cache(args, processor, possible_cache_dir)

    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    
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
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='f1_micro', # TODO f1_macro? mAP? AP?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        remove_unused_columns= False ## TODO
    )

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

    # if not args.eval_only:
    trainer.train(resume_from_checkpoint = args.resume_from_checkpoint)

    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    eval_result = trainer.evaluate(datasets['val'])
    logger.info("* Eval start *")
    logger.info(eval_result)

    logger.info("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    logger.info(test_results)

    if args.snapme_test: # TODO debug
        logger.info("* SNAPMe eval *")
        snapme_result = test_snapme(processor, trainer, args.dataset_cache_path)
        logger.info(snapme_result)

def eval(args):
    """
    Evaluation script for original InstructBlip with T5-xl
    """

    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)
    datasets, _ = load_from_cache(args, processor, possible_cache_dir)
    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    
    logger.info("** Eval only mode **")
    model = load_eval_checkpoint(model, args.eval_checkpoint)

    # if args.eval_checkpoint is None:
    #     raise ValueError("eval_check_point should be provided in eval_only mode")
    
    # logger.info("** Eval only mode **")
    # shard_files = [file for file in os.listdir(args.eval_checkpoint) if file.startswith('pytorch_model') and file.endswith('.bin')]
    # shard_files.sort()

    # checkpoint = {}
    # for shard in shard_files:
    #     shard_path = os.path.join(args.eval_checkpoint, shard)
    #     state_dict_shard = torch.load(shard_path, map_location='cpu')
    #     checkpoint.update(state_dict_shard)

    # model_state_dict = model.state_dict()

    # model_state_dict.update(checkpoint)
    # model.load_state_dict(model_state_dict)
    # logger.info(f"Weight loaded from checkpoint{args.eval_checkpoint}")

    training_args = TrainingArguments( # TODO not necessary
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
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='f1_micro', # TODO f1_macro? mAP? AP?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        remove_unused_columns= False ## TODO
    )

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

    logger.info("* Test start *")
    # datasets['test'] = datasets['test'].select(range(200)) ### TODO
    test_results = trainer.evaluate(datasets['test'])
    logger.info(test_results) # 0.50 나오는 거 보면 제대로 load 된 건데..

    if args.snapme_test: # TODO debug # custom Trainer에 맞게 변형해야 할듯
        logger.info("* SNAPMe eval *")
        snapme_result = test_snapme(processor, trainer, args.dataset_cache_path, gen_mode=args.generate_mode)
        logger.info(snapme_result)


if __name__ == '__main__':
    args = parse_args()

    ###
    args.batch_size = 64
    # args.training_samples = 100
    # args.eval_samples = 100
    # args.eval_steps = 5
    args.project_name = 'temp'
    args.eval_only = True
    args.snapme_test = False
    args.eval_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/original_instructBlip_t5_recipe1m/best'
    args.dataset_name = 'recipe1m'
    # args.project_name = 'temp'
    # args.logging_steps = 50
    ###

    setup_logger(args)
    pretty_print(args)

    if args.eval_only:
        eval(args)
    else:
        train(args)
