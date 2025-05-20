import os
import json
from PIL import Image

from transformers.utils import logging
import torch
from datasets import Dataset
import pandas as pd

logger = logging.get_logger(__name__)

def test_snapme(processor, trainer, dataset_cache_path, test_beverage=True, gen_mode=False):

    possible_cache_dir = os.path.join(dataset_cache_path, 'snapme')
    if os.path.exists(possible_cache_dir):
            logger.info("Load SNAPMe from cache")
            snapme_dataset = Dataset.load_from_disk(possible_cache_dir)
    else:
        logger.info("Create SNAPMe")

        with open('/nfs_share2/shared/from_donghee/snapme/snapme_processed.json', 'r') as f:
            labels = json.load(f)

        id2gt = {entry['id']: entry['gt_int'] for entry in labels}
        id2gt_string = {entry['id']: entry['gt'] for entry in labels} # ingredients in string [milk, sugar, flour...]

        samples = []
        base_path = '/nfs_share2/shared/from_donghee/snapme/snapme_mydb'

        for filename in os.listdir(base_path):
            file_path = os.path.join(base_path, filename)
            if filename.split('/')[-1] in id2gt: 
                try:
                    # with Image.open(file_path) as img:
                    samples.append({
                        'image_path': file_path,
                        'labels': id2gt[filename.split('/')[-1]],
                        'image_id': filename.split('/')[-1],
                        'ingredients': ', '.join(id2gt_string[filename.split('/')[-1]])
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        def process_snapme(examples): # beverage_list
            prompt = 'Question: What are the ingredients I need to make this food? Answer:'

            images, labels, beverage, ingrs = [], [], [], []

            for img_path, label, ingr in zip(examples['image_path'], examples['labels'], examples['ingredients']):
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except:
                    print("Error")
                    continue

                gt = torch.tensor([0]*1488, dtype=torch.float32) ## TODO float32?
                gt[label] = 1
                labels.append(gt)

                img_id = img_path.split("/")[-1]
                is_beverage = 1 if img_id in beverage_list else 0
                beverage.append(is_beverage)

                ingrs.append(ingr)

            assert len(images) == len(labels)

            inputs = processor(images=images, text=[prompt]*len(images), return_tensors='pt')
            inputs['labels'] = torch.stack(labels) # dtype torch.float?
            inputs['is_beverage'] = torch.tensor(beverage)

            temp = processor(text=ingrs, padding='max_length', max_length=128, truncation=True, return_tensors='pt') # ingrs_str
            inputs['ingrs_str'] = temp['input_ids']

            return inputs

        dataset = Dataset.from_pandas(pd.DataFrame(samples))

        with open('/nfs_share2/shared/from_donghee/snapme/snapme_beverage_list.txt', 'r') as file:
            beverage_list = file.read().splitlines()
        beverage_list = set(beverage_list)

        snapme_dataset = dataset.map(process_snapme, batched=True)
        snapme_dataset.save_to_disk(possible_cache_dir)
        logger.info(f"* Save SNAPMe dataset to {possible_cache_dir}") 
    
    if gen_mode:
        # ingredient_ids_one_hot, labels (ingredients list[str])
        snapme_dataset = snapme_dataset.rename_column('labels', 'ingredient_ids_one_hot')
        snapme_dataset = snapme_dataset.rename_column('ingrs_str', 'labels')
        snapme_dataset = snapme_dataset.remove_columns(['ingredients'])

    else:
        snapme_dataset = snapme_dataset.remove_columns(['ingrs_str', 'ingredients'])
 
    snapme_subset = snapme_dataset.remove_columns(['image_path', 'image_id', 'is_beverage'])
    test_result = trainer.evaluate(snapme_subset, metric_key_prefix='snapme') # snapme_f1_micro, snapme_iou_micro
    logger.info(test_result)

    if test_beverage: 
        logger.info("* Eval on two gropus: beverage, non-beverage *")
        
        snapme_dataset_beverage = snapme_dataset.filter(lambda example: example['is_beverage'] == 1) # TODO save disk
        snapme_dataset_beverage = snapme_dataset_beverage.remove_columns(['image_path', 'image_id', 'is_beverage'])
        
        snapme_dataset_non_beverage = snapme_dataset.filter(lambda example: example['is_beverage'] == 0)
        snapme_dataset_non_beverage = snapme_dataset_non_beverage.remove_columns(['image_path', 'image_id', 'is_beverage'])
        
        beverage_result = trainer.evaluate(snapme_dataset_beverage, metric_key_prefix='snapme_beverage')
        non_beverage_result = trainer.evaluate(snapme_dataset_non_beverage, metric_key_prefix='snapme_non_beverage')
        
        logger.info(f"* SNAPMe beverage result: {beverage_result}, non-beverage result: {non_beverage_result}")
    else:
        beverage_result, non_beverage_result = None, None
    
    return test_result, beverage_result, non_beverage_result