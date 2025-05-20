import os
from PIL import Image
import json
import argparse
import pickle
import numpy as np

from matplotlib import pyplot as plt
import textwrap
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration, BERTInstructBlipForConditionalGeneration
from transformers import InstructBlipProcessor,  InstructBlipConfig
from transformers import AutoModel, AutoTokenizer
from transformers.trainer_utils import EvalPrediction
import torch

from data.utils import Vocabulary, to_one_hot

def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='BERT')
    parser.add_argument('--check_point', type=str, default='best')
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
        default=None,
        choices=['bert-large-uncased', 'bert-base-uncased'],
        help="Specifies the BERT model to use. Choose from 'bert-large-uncased', "
            "or 'bert-base-uncased'."
    )

    args = parser.parse_args()
    
    return args

def setup_path(args):
    # load best model
    args.model_dir = os.path.join("./outputs", args.project_name, args.check_point)
    args.model_path = os.path.join(args.model_dir, 'pytorch_model.bin')
    args.qformer_tokenizer_dir = os.path.join(args.model_dir, 'qformer_tokenizer')

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f'model directory does not exist: {args.model_dir}')
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'checkpoint file does not exist: {args.model_path}')
    
    if not os.path.exists(args.qformer_tokenizer_dir):
        print(f'qformer tokenizer directory does not exist: {args.qformer_tokenizer_dir}. Using default tokenizer')

def load_model_tokenizer(args):
    """
    Load a pretrained ViT and LLM model while update qformer with the finetuned parameters
    """
    if args.bert_name is not None:
        model = BERTInstructBlipForConditionalGeneration(args.bert_name)
    else:
        model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    
    qformer_state_dict = torch.load(args.model_path)
    pretrained_state_dict = model.state_dict()
    pretrained_state_dict.update(qformer_state_dict)
    model.load_state_dict(pretrained_state_dict)

    if args.check_point == 'best':
        qformer_tokenizer = AutoTokenizer.from_pretrained(args.qformer_tokenizer_dir)
        llm_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        vision_processor = InstructBlipProcessor.from_pretrained(args.model_name).image_processor
    else:
        print("** Loading checkpoint from ", args.model_path)
        processor = InstructBlipProcessor.from_pretrained(args.model_name)
        llm_tokenizer = processor.tokenizer
        qformer_tokenizer = processor.qformer_tokenizer
        vision_processor = processor.image_processor

    return model, llm_tokenizer, qformer_tokenizer, vision_processor

def retrieve_recipe1m(test_dict):
    dataset = pickle.load(open('/nfs_share2/shared/from_donghee/recipe1m_data/recipe1m_test.pkl', 'rb'))
    ingrs_vocab = pickle.load(open('/nfs_share2/shared/from_donghee/recipe1m_data/recipe1m_vocab_ingrs.pkl', 'rb'))
    max_num_labels = 20
    test_ids = set([entry['id'] for entry in test_dict])

    id2dataset = dict()
    for entry in dataset:
        if entry['id'] in test_ids:
            id2dataset[entry['id']] = entry
    # del dataset

    all_gt = []
    for entry in test_dict:
        sample = id2dataset[entry['id']]
        # ingredient integer
        labels = sample['ingredients']
        ilabels_gt = np.ones(max_num_labels) * ingrs_vocab('<pad>')
        pos = 0

        true_ingr_recipe_idxs = []
        for i in range(len(labels)):
            true_ingr_recipe_idxs.append(ingrs_vocab(labels[i]))

        for i in range(max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_recipe_idx = ingrs_vocab(label)
            if label_recipe_idx not in ilabels_gt:
                ilabels_gt[pos] = label_recipe_idx
                pos += 1

        ilabels_gt[pos] = ingrs_vocab('<end>')
        all_gt.append(ilabels_gt)
        # ingr_gt_int = torch.from_numpy(ilabels_gt).long()
    
    all_gt = torch.tensor(np.array(all_gt)).long()
    # all_gt_one_hot = to_one_hot(all_gt)

    return all_gt


def demo_dataset(vision_processor, llm_tokenizer, qformer_tokenizer, device):
    """
    return sample, labels (tensor, not one-hot vector)
    """
    
    with open('data/demo_test_images/demo_test.json', 'r') as f:
        test_json = json.load(f)

    labels = retrieve_recipe1m(test_json)

    # ground_truths = [', '.join(ingr['text'] for ingr in entry['ingredients']) for entry in labels]


    images = []
    for filename in os.listdir('data/demo_test_images'):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join('data/demo_test_images', filename))
            images.append(img)

    prompts = ['Question: What are the ingredients I need to make this food? Answer:']*len(images)

    llm_prompt = llm_tokenizer(prompts, padding='longest', return_tensors='pt').to(device)
    qformer_prompt = qformer_tokenizer(prompts, padding='longest', return_tensors='pt').to(device)
    pixel_values = vision_processor(images, return_tensors='pt').pixel_values.to(device)

    sample = {
        'images': images,
        'pixel_values': pixel_values,
        'input_ids': llm_prompt.input_ids,
        'attention_mask': llm_prompt.attention_mask,
        'qformer_input_ids': qformer_prompt.input_ids,
        'qformer_attention_mask': qformer_prompt.attention_mask
    }

    return sample, labels

def pretty_print(outputs, images, labels, llm_tokenizer, save_path):
    """
    Display images with ground truth labels and generated predictions.
    """
    def wrap_text(text, width):
        """Wrap text for better display in plots."""
        return '\n'.join(textwrap.wrap(text, width=width))
    
    predictions = torch.sigmoid(torch.tensor(outputs.logits))
    mask = predictions > 0.5
    pred_idx = torch.nonzero(mask, as_tuple=False)

    with open('data/ingr_cluster.json', 'r') as f:
        ingr_cluster = json.load(f)

    all_pred_ingrs = [[] for _ in range(predictions.shape[0])]
    for idx in pred_idx:
        batch_idx, label_idx = idx.tolist()
        all_pred_ingrs[batch_idx].append(ingr_cluster[str(label_idx)])

    ingr_labels = [[] for _ in range(predictions.shape[0])]
    for idx in labels:
        batch_idx, label_idx = idx.tolist()
        ingr_labels[batch_idx].append(ingr_cluster[str(label_idx)])

    num_images = len(images)
    fig, axs = plt.subplots(num_images, 1, figsize=(15, 20 * num_images))

    if num_images == 1:
        axs = [axs]
    
    for i, (img, label, prediction) in enumerate(zip(images, ingr_labels, all_pred_ingrs)):
        axs[i].imshow(img)
        axs[i].axis('off')
        title_text = f"- Ground truth: {wrap_text(label, 80)}\n- Prediction: {wrap_text(prediction, 80)}"
        axs[i].text(0.5, -0.1, title_text, transform=axs[i].transAxes, fontsize=20, 
                    verticalalignment='top', horizontalalignment='center') 

    plt.tight_layout()
    filename = os.path.join(save_path, 'demo.png')
    plt.savefig(filename, bbox_inches='tight')
    print(f"Figure saved at {filename}")
    plt.show()

def compute_metrics(pred):
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

    print(f'* Evaluation result: {result}')

    return result

def test(args):
    """
    Temporal test with demo data
    """

    ## TODO !!! - load....
    model, llm_tokenizer, qformer_tokenizer, vision_processor = load_model_tokenizer(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    print("Load done")

    inputs, labels = demo_dataset(vision_processor, llm_tokenizer, qformer_tokenizer, device)
    images = inputs.pop('images', None)

    # outputs = model.generate(
    #     **inputs,
    #     do_sample=False,
    #     num_beams=5,
    #     max_length=256,
    #     min_length=1,
    #     repetition_penalty=1.5,
    #     length_penalty=1.0,
    #     temperature=1,
    # )
    outputs = model(**inputs)
    labels_one_hot = to_one_hot(labels)
    result = compute_metrics(EvalPrediction(predictions=outputs.logits.cpu(), label_ids=labels_one_hot))

    pretty_print(outputs, images, labels, llm_tokenizer, save_path=args.model_dir)
    print()


if __name__ == '__main__':

    args = parse_args()
    ##
    args.project_name = 'BERT'
    args.check_point = 'checkpoint-32240'
    args.bert_name = 'bert-large-uncased'
    ##
    setup_path(args)
    test(args)