import json
# import logging
from transformers.utils import logging
import os
import argparse

import torch
from transformers import TrainingArguments, Trainer, TrainerCallback, ViTConfig, AutoImageProcessor
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.dataset import load_datasets
from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from datasets import load_dataset, DatasetDict

from model.modeling_vit import MyViTForImageClassification
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre, compute_metrics_acc, compute_metrics_f1

# TODO
# 5. log only main process /

logger = logging.get_logger(__name__)

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='ViT_only')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')
    parser.add_argument('--dataset_name', type=str, default='recipe1m', choices=['recipe1m', 'mnist', 'cifar10', 'cifar100'], help='Hugging face built-in datasets or Recipe1M')
    parser.add_argument('--dataset_cache_path', type=str, default='/home/donghee/huggingface_data_cache', help='local dataset cache directory')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--eval_steps', type=int, default=500) 
    parser.add_argument('--logging_steps', type=int, default=50) 
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--load_from_cache_file', type=bool, default=True, help='load dataset from huggingface cache')
    parser.add_argument('--train_vit', type=bool, default=False, help='train ViT')
    parser.add_argument('--fine_label', type=bool, default=False, help='if True, use fine labels for classification')
    parser.add_argument('--eval_split_ratio', type=float, default=0.1, help='split ratio for validation set')
    parser.add_argument('--generate_mode', type=bool, default=False)
    parser.add_argument('--classification_mode', type=bool, default=True)

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='google/vit-base-patch16-224-in21k',
        choices=['google/vit-base-patch16-224-in21k', 'google/vit-large-patch32-224-in21k', 'google/vit-huge-patch14-224-in21k'],
        help="Specifies the model to use. Choose from 'google/vit-base-patch16-224-in21k' (default), 'google/vit-large-patch32-224-in21k', 'google/vit-huge-patch14-224-in21k'. "
    )
    
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()
    
    # args.output_dir= os.path.join("./outputs", args.project_name)
    # args.logging_dir = os.path.join('./logs', args.project_name)

    return args


def train(args):
    
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    # TODO idenity multi-label classification
    multi_classification = True if args.dataset_name == 'recipe1m' else False
    ##
    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)

    if args.dataset_name == 'recipe1m':
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            datasets = remove_unused_columns(datasets, args.generate_mode, vit_only=True)

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
            datasets.save_to_disk(possible_cache_dir)
            datasets = remove_unused_columns(datasets, args.generate_mode, vit_only=True)
            logger.info(f"* Save dataset to {possible_cache_dir}") 
        
        num_labels = len(datasets['train'][0]['labels']) ## TODO -1 # padding
            
    else: # TODO # compatible
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            # TODO make class_names compatible to other datasets (coarse label, fine label..)
            class_names = datasets['train'].features['coarse_label'].names if not args.fine_label else datasets['train'].features['fine_label'].names
            num_labels = len(class_names)
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
            
            os.makedirs(possible_cache_dir)
            datasets.save_to_disk(possible_cache_dir)
            logger.info(f"* Save dataset to {possible_cache_dir}") 

    config = ViTConfig()
    model = MyViTForImageClassification(config, model_name=args.model_name, vit_freeze=args.vit_freeze, num_labels=num_labels, multi_classification=multi_classification)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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
        metric_for_best_model='max_f1' if multi_classification else 'accuracy',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
    )
    
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'][:32],
        eval_dataset=datasets['val'],
        # compute_metrics=compute_metrics_thre if multi_classification else compute_metrics_acc
        compute_metrics = compute_metrics_f1
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # model.save_pretrained_final(os.path.join(args.output_dir, 'best')) ## TODO

    # eval_result = trainer.evaluate()
    # logger.info("EVAL")
    # logger.info(eval_result)

    logger.info("* Test start *")
    test_results = trainer.evaluate(datasets['test'], metric_key_prefix='test')
    logger.info(test_results)


if __name__ == '__main__':
    args = parse_args()

    ####
    # args.training_samples = 128
    args.epochs = 1
    args.batch_size = 32
    args.eval_batch_size = 256
    args.vit_freeze = True
    args.project_name = 'temp' # 'vit_only_recipe1m'
    # args.eval_steps = 5
    # args.eval_samples = -1
    # args.test_samples = 10000
    args.model_name = 'google/vit-huge-patch14-224-in21k'
    args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/vit_only_recipe1m/checkpoint-4500'
    ####

    setup_logger(args)
    pretty_print(args)

    train(args)
