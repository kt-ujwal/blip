# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This script is based on code from Inversecooking by Facebook
"""

import os
import pickle
import random
import json
import lmdb
import nltk
# import logger
from transformers.utils import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import Features, Value, Array3D, Sequence
from datasets import Dataset as hf_Dataset
from transformers import DataCollatorWithPadding

from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from datasets import DatasetDict, load_dataset

# TODO better way
logger = logging.get_logger(__name__)

class Recipe1MDataset(Dataset):
    """
    This Dataset class is designed to work wirh LMDB files, assuming that the images are pre-stored in a LMDB database.
    It DOES NOT support loading images directly from JPEG files.
    """
    def __init__ (self, processor, data_dir, split, max_num_labels, max_num_samples=-1):
        
        self.split = split

        # load 3 files: lmdb, recipe1m pkl, vocab file
        self.image_file = lmdb.open(os.path.join(data_dir, 'lmdb_'+ split), max_readers=1, readonly=True, lock = False, readahead=False, meminit=False)
        self.dataset = pickle.load(open(os.path.join(data_dir, 'recipe1m_'+split+'.pkl'), 'rb'))
        self.ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'recipe1m_vocab_ingrs.pkl'), 'rb'))
        
        self.processor = processor

        self.ids = [] 
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        self.maxnumims = 5
        self.max_num_labels = max_num_labels
        if max_num_samples != -1: 
            # # randomly sample subset
            # random.shuffle(self.ids)
            ## no longer random - for caching
            self.ids = self.ids[:max_num_samples]


    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient
    
    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)
    
    def __len__(self):
        return len(self.ids)
    
    def to_list(self):
        # Convert the dataset to a list of dictionaries
        data_list = []
        for idx in range(len(self)):
            data_point = self.__getitem__(idx)
            data_list.append(data_point)
        return data_list
    
    def __getitem__(self, index):
        
        current_id = self.ids[index]
        sample = self.dataset[current_id]
        recipe_id = sample['id']
        # captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims]
        if self.split == 'train':
            idx = np.random.randint(0, len(paths))
        else:
            idx = 0
        path = paths[idx]

        try:
            with self.image_file.begin(write=False) as txn:
                image = txn.get(path.encode())
                image = np.fromstring(image, dtype=np.uint8)
                image = np.reshape(image, (256, 256, 3))
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            # pixel_values = self.processor(images=image)['pixel_values'][0] #### TODO 
        except:
            # print ("Image recipe_id not found in lmdb. Loading jpeg file...")
            # image = Image.open(os.path.join(self.root, path[0], path[1],
            #                                 path[2], path[3], path)).convert('RGB')
            logger.info("Not able to load lmdb file")
            pixel_values = None

        # idx = index

        ingr_gt = sample['ingredients']
        # random.shuffle(ingr_gt) ## shuffle ingredients
        ingr_gt.sort()  # TODO current cache file is random..
        ingr_gt = ", ".join(ingr_gt)
        ingr_gt = ingr_gt.replace('_',' ')
        # text_output = self.processor(text=ingr_gt)
        # input_ids = text_output.input_ids
        # attention_mask = text_output.attention_mask
        # label_ids = text_output.input_ids ### TODO

        # labels = self.dataset[self.recipe_ids[recipe_idx]]['ingredients']
        title = sample['title']
        title = " ".join(title)

        # ingredient integer
        labels = sample['ingredients']
        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        true_ingr_recipe_idxs = []

        true_ingr_recipe_idxs = []
        for i in range(len(labels)):
            true_ingr_recipe_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_recipe_idx = self.ingrs_vocab(label)
            if label_recipe_idx not in ilabels_gt:
                ilabels_gt[pos] = label_recipe_idx
                pos += 1

        ilabels_gt[pos] = self.ingrs_vocab('<end>')
        ingr_gt_int = torch.from_numpy(ilabels_gt).long()

        question = 'Question: What are the ingredients I need to make this food? Answer:'
        # text_input = self.processor(text=question)
        # qformer_input = self.qformer_tokenizer(question)
        # qformer_input_ids = text_input.qformer_input_ids
        # qformer_attention_mask = text_input.qformer_attention_mask
        # input_ids = text_input.input_ids
        # attention_mask = text_input.input_ids

        ### TODO
        # llm_input = self.llm_tokenizer(question)
        # input_ids = llm_input.input_ids
        # attention_mask = llm_input.attention_mask
        ####
        # return {
        #     'pixel_values': pixel_values,
        #     'qformer_input_ids': qformer_input_ids,
        #     'qformer_attention_mask': qformer_attention_mask,
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'label_ids': label_ids
        # }

        return {
            'image': image,
            'text_input': question,
            'text_output': ingr_gt,
            'text_input_output': question+ingr_gt,
            'ingredient_int': ingr_gt_int,
            # 'title': title,
            # 'recipe_id': recipe_id,
            # 'img_id': path
        }


def collator(data):
    image_list, input_list, output_list, ingr_list, title_list, recipe_id_list, path_list = [], [], [], [], [], [], []

    for sample in data:
        if sample['image'] is not None:
            image_list.append(sample['image'])
            input_list.append(sample['text_input'])
            output_list.append(sample['text_output'])
            ingr_list.append(sample['ingredient_int'])
            title_list.append(sample['title'])
            recipe_id_list.append(sample['recipe_id'])
            path_list.append(sample['img_id'])
    
    return {
        'image' : torch.stack(image_list, dim=0),
        'text_input': input_list,
        'text_output': output_list,
        'ingredient_int': torch.stack(ingr_list, dim=0),
        'title': title_list,
        'recipe_id': recipe_id_list,
        'img_id': path_list
    }


def collate_fn(data):
    
    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    data = [sample for sample in data if sample[0] is not None] ##
    image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.ones(len(captions), max(lengths)).long()*pad_value[0]

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return image_input, targets, ingrs_gt, img_id, path


def recipe1m_generator(processor, data_dir, split, max_num_labels, max_num_samples):
    recipe_dataset = Recipe1MDataset(processor, data_dir, split, max_num_labels, max_num_samples)
    for i in range(len(recipe_dataset)):
        item = recipe_dataset[i]
        image_array = np.array(item['image'])  # Convert PIL Image to numpy array
        image_array = image_array.transpose((2, 0, 1))  # Convert HWC to CHW format if needed
        yield {
            'image': image_array,
            'text_input': item['text_input'],
            'text_output': item['text_output'],
            'text_input_output': item['text_input_output'],
            'ingredient_int': item['ingredient_int'].tolist()  # Convert torch tensor to list
        }

def load_image_datasets(
    image_processor,
    data_dir, 
    training_samples,
    eval_samples, 
    pre_map,
    load_from_cache_file=True,
    max_num_labels=20,
):
    if not pre_map:
        logger.info('batches of data are processed on-the-fly')

    if training_samples != -1:
        logger.info(f"Limiting the number of training samples to {training_samples}")
    
    def process_data(example):
        sample = dict()
        
        sample['pixel_values'] = image_processor(np.array(example['image'])).pixel_values
        sample['labels'] = to_one_hot(torch.tensor(example['ingredient_int']), remove_pad = True)

        return sample

    datasets = dict()
    features = Features({
        'image': Array3D(dtype="float32", shape=(3, 256, 256)),  # Assuming image converted to 3D array
        'text_input': Value(dtype='string'),
        'text_output': Value(dtype='string'),
        'text_input_output': Value(dtype='string'),
        'ingredient_int': Sequence(feature=Value(dtype='int64'))
    })

    for split in ['train', 'val', 'test']:
        max_num_samples = eval_samples if split in ['val', 'test'] else training_samples

        # Define a lambda function that includes the arguments
        generator_function = lambda split=split, max_num_samples=max_num_samples: recipe1m_generator(
            processor=None,
            data_dir=data_dir,
            split=split,
            max_num_labels=max_num_labels,
            max_num_samples=max_num_samples,
        )

        gen_dataset = hf_Dataset.from_generator(
            generator = generator_function,
            features = features,
        )
        if pre_map:
            datasets[split] = gen_dataset.map(process_data, batched=True, load_from_cache_file=load_from_cache_file)
        else:
            datasets[split] = gen_dataset

    del gen_dataset

    return datasets
    

def llava_load_datasets(
        processor,
        data_dir,
        training_samples = -1,
        eval_samples = 1500,
        test_samples = -1,
        max_num_labels = 20,
        pre_map = False,
        decoder_only = True,
        encoder_only = False,
        load_from_cache_file = True,
    ) -> Dict[str, hf_Dataset]:

    if not pre_map:
        logger.info('batches of data are processed on-the-fly')

    if training_samples != -1:
        logger.info(f"Limiting the number of training samples to {training_samples}")
    
    if not pre_map:
        logger.info('batches of data are processed on-the-fly')

    if training_samples != -1:
        logger.info(f"Limiting the number of training samples to {training_samples}")
    
    def process_data(example):
        if decoder_only:
            inputs = processor(text=example['text_input_output'], images=np.array(example['image']), return_tensors='pt', padding='max_length', max_length=128)
            inputs['labels'] = processor(text=example['text_output'], return_tensors='pt', padding='max_length', max_length=128).input_ids
            # TODO what if inputs['input_ids'] have different length?
        return inputs

    datasets = dict()
    features = Features({
        'image': Array3D(dtype="float32", shape=(3, 256, 256)),  # Assuming image converted to 3D array
        'text_input': Value(dtype='string'),
        'text_output': Value(dtype='string'),
        'text_input_output': Value(dtype='string'),
        'ingredient_int': Sequence(feature=Value(dtype='int64'))
    })

    for split in ['train', 'val', 'test']:
        # max_num_samples = eval_samples if split in ['val', 'test'] else training_samples
        if split == 'train':
            max_num_samples = training_samples
        elif split == 'val':
            max_num_samples = eval_samples
        else:
            max_num_samples = test_samples

        # Define a lambda function that includes the arguments
        generator_function = lambda split=split, max_num_samples=max_num_samples: recipe1m_generator(
            processor=processor,
            data_dir=data_dir,
            split=split,
            max_num_labels=max_num_labels,
            max_num_samples=max_num_samples,
        )

        gen_dataset = hf_Dataset.from_generator(
            generator = generator_function,
            features = features,
        )
        if pre_map:
            datasets[split] = gen_dataset.map(process_data, batched=True, load_from_cache_file=load_from_cache_file)
        else:
            datasets[split] = gen_dataset

    del gen_dataset

    return datasets
    

def load_datasets(
        processor,
        data_dir, 
        training_samples=-1, 
        eval_samples=1500,
        test_samples=-1, 
        max_num_labels=20, 
        pre_map=False, 
        decoder_only=False, 
        encoder_only=False, 
        load_from_cache_file=True,
    ) -> Dict[str, hf_Dataset]:
    '''
    :param data_dir: recipe1M dataset directory containing 
        1. lmdb dir (lmdb_train, lmdb_val, lmdb_train)
        2. recipe1m_train,val,test pkl - contains ingredient list, recipe instruction, title etc
        3. recipe1m_vocab_ingrs.pkl
    :param training_samples[Optional]: limiting the number of training samples. All training samples are used by default

    Limits the number of validation and test samples to 1500
    '''
    if not pre_map:
        logger.info('batches of data are processed on-the-fly')

    if training_samples != -1:
        logger.info(f"Limiting the number of training samples to {training_samples}")
    
    def process_data(example):
        # processor.tokenizer.padding_side = 'right'
        # processor.qformer_tokenizer.padding_side = 'right'
        sample = dict()
        
        prompt = processor(
            images=np.array(example['image']),
            text=example['text_input'],
            return_tensors='pt',
            truncation=True,
            padding=False,
        )

        if encoder_only:
            sample = prompt
            sample['label'] = to_one_hot(torch.tensor(example['ingredient_int']))
            return sample

        output = processor(
            text=example['text_output'], 
            padding='max_length',
            max_length=128,
            truncation=True,
            # padding=False,
            return_tensors='pt') # ingredient list

        sample['pixel_values'] = prompt.pixel_values
        sample['qformer_input_ids'] = prompt.qformer_input_ids
        sample['qformer_attention_mask'] = prompt.qformer_attention_mask
        # sample['label_ids'] = output.input_ids
        sample['labels'] = to_one_hot(torch.tensor(example['ingredient_int'])) ## TODO label ids? label?

        # sample['ingredient_ids'] = to_one_hot(torch.tensor(example['ingredient_int']))

        if decoder_only:
            prompt_plus_output = processor(
                text = example['text_input_output'],
                padding='max_length',
                max_length=128,
                truncation=True,
                # padding=False,
                return_tensors='pt',
            )

            sample['input_ids'] = prompt_plus_output.input_ids
            sample['attention_mask'] = prompt_plus_output.attention_mask
            del prompt_plus_output
        
        else:
            # for encoder-decoder model like t5
            sample['input_ids'] = prompt.input_ids
            sample['attention_mask'] = prompt.attention_mask
            sample['decoder_input_ids'] = output.input_ids
            sample['decoder_attention_mask'] = output.attention_mask

        del prompt, output
        return sample

    datasets = dict()
    features = Features({
        'image': Array3D(dtype="float32", shape=(3, 256, 256)),  # Assuming image converted to 3D array
        'text_input': Value(dtype='string'),
        'text_output': Value(dtype='string'),
        'text_input_output': Value(dtype='string'),
        'ingredient_int': Sequence(feature=Value(dtype='int64'))
    })

    for split in ['train', 'val', 'test']:
        # max_num_samples = eval_samples if split in ['val', 'test'] else training_samples
        if split == 'train':
            max_num_samples = training_samples
        elif split == 'val':
            max_num_samples = eval_samples
        else:
            max_num_samples = test_samples

        # Define a lambda function that includes the arguments
        generator_function = lambda split=split, max_num_samples=max_num_samples: recipe1m_generator(
            processor=processor,
            data_dir=data_dir,
            split=split,
            max_num_labels=max_num_labels,
            max_num_samples=max_num_samples,
        )

        gen_dataset = hf_Dataset.from_generator(
            generator = generator_function,
            features = features,
        )
        if pre_map:
            datasets[split] = gen_dataset.map(process_data, batched=True, load_from_cache_file=load_from_cache_file)
        else:
            datasets[split] = gen_dataset

    del gen_dataset

    return datasets

def load_from_cache(args, processor, possible_cache_dir, remove_columns: List[str]=None):
    """
    Try loading from cache, otherwise generate and map dataset
    return datasets, num_labels
    """
    if args.dataset_name in {'recipe1m', 'recipe1m_sorted'}:
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            datasets = remove_unused_columns(datasets, args.generate_mode)
            if remove_columns is not None:
                datasets = datasets.remove_columns(remove_columns)
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
            logger.info(f"* Save dataset to {possible_cache_dir}") 
        
        num_labels = len(datasets['train'][0]['labels']) ## TODO -2 # 0: <end>, 1487: <pad>
            
    else:
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
    
    return datasets, num_labels

# class CustomDataCollator:
#     def __init__(self, processor, decoder_only):
#         self.qformer_data_collator = DataCollatorWithPadding(processor.qformer_tokenizer)
#         self.llm_data_collator = DataCollatorWithPadding(processor.tokenizer)
#         self.decoder_only = decoder_only
    
#     def __call__(self, examples):
#         examples['qformer_input_ids'] = self.qformer_data_collator(examples['qformer_input_ids'])
#         examples['qformer_attention_mask'] = self.qformer_data_collator(examples['qformer_attention_mask'])
#         examples['input_ids'] = self.llm_data_collator(examples['input_ids'])
#         examples['attention_mask'] = self.llm_data_collator(examples['attention_mask'])

#         if not self.decoder_only:
#             examples['decoder_input_ids'] = self.llm_data_collator(examples['decoder_input_ids'])
#             examples['decoder_attention_mask'] = self.llm_data_collator(examples['decoder_attention_mask'])
        
#         return examples

# def load_datasets_for_distributed(processor, data_dir, rank, world_size, training_samples=-1, max_num_labels=20, pre_map=False, decoder_only=False):
    
#     datasets = load_datasets(processor, data_dir, training_samples, max_num_labels, pre_map, decoder_only)

#     for split in datasets.keys():
#         # Calculate start and end indices for each process
#         dataset = datasets[split]
#         total_size = len(dataset)
#         per_process_size = total_size // world_size
#         start = rank * per_process_size
#         end = start + per_process_size if rank != world_size - 1 else total_size

#         # Slice the dataset for the current process
#         datasets[split] = dataset.select(range(start, end))
#         logger.info(f"* Size of {split} dataset for each processor: {len(datasets[split])}")

#     return datasets

# class Recipe1M_Collator(DataCollatorWithPadding):
#     def __init__(self, processor):
#         super().__init__(tokenizer=processor.tokenizer, padding=True)
#         self.processor = processor

#     def __call__(self, batch):
#         # Initialize lists to store processed fields
#         pixel_values = []
#         qformer_input_ids = []
#         qformer_attention_mask = []
#         input_ids = []
#         attention_mask = []
#         label_ids = []
#         ingredient_int = []

#         # Process each item in the batch
#         for item in batch:
#             # Process the image and text input
#             encoding = self.processor(
#                 images=np.array(item['image']), 
#                 text=item['text_input'], 
#                 padding='longest', 
#                 return_tensors='pt'
#             )

#             # Process the text output for labels
#             labels = self.processor(
#                 text=item['text_output'], 
#                 padding='longest', 
#                 return_tensors='pt'
#             ).input_ids

#             # Append processed items to their respective lists
#             pixel_values.append(encoding.pixel_values.squeeze())
#             qformer_input_ids.append(encoding.qformer_input_ids.squeeze())
#             qformer_attention_mask.append(encoding.qformer_attention_mask.squeeze())
#             input_ids.append(encoding.input_ids.squeeze())
#             attention_mask.append(encoding.attention_mask.squeeze())
#             label_ids.append(labels.squeeze())
#             ingredient_int.append(torch.tensor(item['ingredient_int']).long())

#         # Combine the lists into a batch dictionary
#         processed_batch = {
#             'pixel_values': torch.stack(pixel_values),
#             'qformer_input_ids': torch.stack(qformer_input_ids),
#             'qformer_attention_mask': torch.stack(qformer_attention_mask),
#             'input_ids': torch.stack(input_ids),
#             'attention_mask': torch.stack(attention_mask),
#             'labels': torch.stack(label_ids),
#             'ingredient_int': torch.stack(ingredient_int)
#         }

#         # Utilize parent class method for padding
#         return super().__call__(processed_batch)

