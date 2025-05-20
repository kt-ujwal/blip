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
from evaluate import load

# from torchvision.datasets import VOCDetection
from data.dataset import load_datasets
from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from model.modeling_instructblip import QT5InstructBlipForClassification
from common.dist_utils import init_distributed_mode
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre, compute_metrics_acc, ComputeMetricsF1

from transformers import T5ForSequenceClassification
from transformers.trainer_utils import EvalPrediction
from datasets import load_dataset, DatasetDict, Dataset
from data.utils import to_one_hot, get_cache_file_name
from common.snapme import test_snapme

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
    parser.add_argument('--eval_checkpoint', type=str, default=None)

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
    parser.add_argument('--eval_only', type=bool, default=True, help='if True, no train')
    parser.add_argument('--beverage_test', type=bool, default=True, help='if True, compare on SNAPMe beverage group and non-beverage group. snapme should be set to True')
    parser.add_argument('--snapme', type=bool, default=True, help='if True, no train run, only eval on SNAPMe')

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
    multi_classification = True if args.dataset_name == 'recipe1m' or args.snapme else False
    ##
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)

    if args.dataset_name in {'recipe1m', 'recipe1m_sorted'}:
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            datasets = remove_unused_columns(datasets, args.generate_mode)

        else:
            logger.info("* Recipe1M mapping start")
            datasets = load_datasets( 
                processor=processor, 
                data_dir=args.dataset_path, 
                training_samples=args.training_samples,
                eval_samples=args.eval_samples, 
                pre_map=args.pre_map,
                decoder_only=args.decoder_only
            )
            datasets = DatasetDict(datasets)
            logger.info("* Recipe1M saving start")
            os.makedirs(possible_cache_dir)
            datasets.save_to_disk(possible_cache_dir) # TODO put one line down
            datasets = remove_unused_columns(datasets, args.generate_mode)
            logger.info(f"* Save dataset to {possible_cache_dir}") # TODO log cols
        
        num_labels = len(datasets['train'][0]['labels']) ## TODO -1 # padding
        logger.info(f"number of labels: {num_labels}")
            
    else:
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            # TODO make class_names compatible to other datasets (coarse label, fine label..)
            # class_names = datasets['train'].features['coarse_label'].names if not args.fine_label else datasets['train'].features['fine_label'].names
            class_names = datasets['train'].features['labels'] ## TODO debug
            num_labels = len(class_names)
            datasets = remove_unused_columns(datasets, args.generate_mode, classification_mode=True)
        else:
            # TODO optimize
            if args.dataset_name == 'cifar100_fine_label':
                datasets = load_dataset('cifar100')    
            else:
                datasets = load_dataset(args.dataset_name)
            # TODO make class_names compatible to other datasets (coarse label, fine label..)
            class_names = datasets['train'].features['coarse_label'].names if not args.fine_label else datasets['train'].features['fine_label'].names
            num_labels = len(class_names)
            class_names = ", ".join(class_names).replace('_', ' ')

            def preprocess_data(examples):
                text_input = [f'Identify the main object in the image from the following categories: {class_names}']*len(examples['img'])
                inputs = processor(
                    images = examples['img'],
                    text = text_input,
                    return_tensors='pt',
                ) # input_ids, attention_mask, qformer_iput_ids, qformer_attention_mask, pixel_values

                inputs['labels'] = to_one_hot(examples['coarse_label'] if not args.fine_label else examples['fine_label'], num_classes = num_labels, remove_pad=False) # one-hot labels
                
                return inputs
            
            if len(datasets) == 2: # no eval split
                eval_split_ratio = args.eval_split_ratio # 0.1
                train_test_split = datasets["train"].train_test_split(test_size=eval_split_ratio)
                datasets = DatasetDict({
                    'train': train_test_split['train'],
                    'val': train_test_split['test'],  # new validation set
                    'test': datasets['test']
                })
            
            assert len(datasets) == 3
            datasets = datasets.map(preprocess_data, batched=True)
            datasets = remove_unused_columns(datasets, args.generate_mode, classification_mode=True)
            
            os.makedirs(possible_cache_dir)
            datasets.save_to_disk(possible_cache_dir)
            logger.info(f"* Save dataset to {possible_cache_dir}") 

    # processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    
    # model = QT5InstructBlipForClassification.from_pretrained(args.model_name)

    # model.learnable_query_init(num_query=args.num_query, num_labels=num_labels, freeze_qformer=args.freeze_qformer, multi_classification=multi_classification)
    
    # if args.eval_only:
    #     shard_files = [file for file in os.listdir(args.eval_checkpoint) if file.startswith('pytorch_model') and file.endswith('.bin')]
    #     shard_files.sort()

    #     checkpoint = {}
    #     for shard in shard_files:
    #         shard_path = os.path.join(args.eval_checkpoint, shard)
    #         state_dict_shard = torch.load(shard_path, map_location='cpu')
    #         checkpoint.update(state_dict_shard)

    #     model_state_dict = model.state_dict()

    #     model_state_dict.update(checkpoint)
    #     model.load_state_dict(model_state_dict)
    
    # training_args = TrainingArguments(
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.eval_batch_size if args.eval_batch_size else args.batch_size,
    #     num_train_epochs=args.epochs,
    #     evaluation_strategy="steps",
    #     eval_steps=args.eval_steps, # 500
    #     logging_dir=args.logging_dir,
    #     logging_strategy = 'steps',
    #     logging_steps=args.logging_steps,
    #     output_dir=args.output_dir,
    #     save_strategy="steps",
    #     save_steps=args.eval_steps,
    #     save_total_limit=4,
    #     load_best_model_at_end=True,
    #     metric_for_best_model='max_f1' if multi_classification else 'accuracy', # TODO multi-label classification metrics.. mAP? AP?
    #     greater_is_better=True,
    #     dataloader_num_workers=4,
    #     ddp_find_unused_parameters=False,
    #     save_safetensors=False,
    #     auto_find_batch_size=True, ##
    #     # include_inputs_for_metrics=True,
    #     # remove_unused_columns= False ## TODO
    # )
    
    # trainer = Trainer( 
    #     model=model,
    #     args=training_args,
    #     train_dataset=datasets['train'],
    #     eval_dataset=datasets['val'],
    #     compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    # )

    # if not args.eval_only:
    #     trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    #     # Save
    #     model.save_pretrained_final(os.path.join(args.output_dir, 'best'))
    
    #     logger.info("* Eval start *")
    #     eval_result = trainer.evaluate(datasets['val'])
    #     logger.info(eval_result)

    # logger.info("* Test start *")
    # if multi_classification:
    #     # eval_result = trainer.evaluate(datasets['val'])
    #     # eval_threshold = eval_result['eval_max_threshold']
    #     eval_threshold = 0.3  ## TODO
    #     trainer.compute_metrics = ComputeMetricsF1(threshold=eval_threshold)

    # # datasets['test'] = datasets['test'].select(range(200)) # take subset # TODO
    # # test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    # # logger.info(test_results)

    # # with open(os.path.join(args.output_dir, 'best', 'test_result.json'), 'w') as f:
    # #     json.dump(test_results, f, indent=4)
    # #     logger.info(f"* Save test result to {os.path.join(args.output_dir, 'best', 'test_result.json')}")

    # if args.dataset_name == 'recipe1m' or args.snapme:
    #     logger.info("* SNAPMe eval *")
    #     snapme_result, snapme_beverage_result, snapme_non_beverage_result = test_snapme(processor, trainer, dataset_cache_path=args.dataset_cache_path, test_beverage=args.beverage_test)
    #     logger.info(snapme_result) # TODO debug.. 

    #     if snapme_beverage_result is not None:
    #         snapme_result.update(snapme_beverage_result)
    #         snapme_result.update(snapme_non_beverage_result)

    #     with open(os.path.join(args.output_dir, 'best', 'snapme_result.json'), 'w') as f:
    #         json.dump(snapme_result, f, indent=4)
    #         logger.info(f"* Save SNAPMe test result to {os.path.join(args.output_dir, 'best', 'snapme_result.json')}")


if __name__ == '__main__':
    args = parse_args()

    ###
    args.batch_size = 32 # batchsize 32, num_query 4 : 33GB, batchsize 16 num_query 1: 24GB
    # args.eval_batch_size = 128
    # args.training_samples = 32
    # args.eval_samples = 128
    # args.eval_steps = 10
    # args.logging_steps = 5
    # args.epochs = 1
    args.num_query = 1
    args.project_name = 'temp'
    # args.project_name = 'temp'
    args.eval_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/t5_learnable_query1_recipe1m/best'
    args.dataset_name = 'recipe1m_sorted'
    args.eval_only = True
    ###

    setup_logger(args)
    pretty_print(args)

    train(args)
