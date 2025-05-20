# Training script for InstructBlip Models

This repository provides an approach to training and evaluating different models within the InstructBlip framework, specifically BERT, Qformer, and T5 models. It is designed to handle Recipe1M and SNAPMe dataset.


## Task Description

The primary objective of this repository is to enable the training and evaluation of models for the task of **predicting ingredients from food images**.

## Model configurations

Each model type leverages the InstructBlip framework from Salesforce differently.

### T5 (Original InstructBlip)
- **Original InstructBlip**, using T5 XL version as the backbone.

### BERT
- Replaces the language model of InstructBlip with **BERT Large**.

### Qformer
- Simplifies the architecture by **removing the language model** and attaching a classifier directly to the Qformer, streamlining the process for direct classification tasks.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/donghee1ee/blip.git
cd https://github.com/donghee1ee/blip.git
pip install -r requirements.txt
```

## Usage

To use the script, you need to specify at least the model type (`--model_type`), the project name (`--project_name`), and the dataset path (`--dataset_path`). Here's an example command:

```bash
python run.py --model_type T5 --project_name "MyT5Project" --dataset_path "/path/to/dataset"
```

### Command-Line Arguments

- `--model_type`: Type of model to train (`BERT`, `Qformer`, `T5`).
- `--project_name`: Name of the training project.
- `--dataset_path`: Path containing the dataset.
- `--eval_only`: Set to `True` for evaluation mode. Default is `False`.
- `--snapme_test`: Set to `True` to enable testing on the SNAPMe dataset. Default is `False`.

For a full list of arguments, use:

```bash
python run.py --help
```

## Examples

### Training Examples

#### BERT
```bash
python run.py --model_type BERT --project_name "BERTProject" --dataset_path "/path/to/dataset" --snapme_test=True
```

#### Qformer
```bash
python run.py --model_type Qformer --project_name "QformerProject" --dataset_path "/path/to/dataset" --snapme_test=True
```

#### T5
```bash
python run.py --model_type T5 --project_name "T5Project" --dataset_path "/path/to/dataset" --snapme_test=True
```

### Evaluation Examples

Add `--eval_only=True` in any of the above commands to switch to evaluation mode. For example:

#### BERT Evaluation
```bash
python run.py --model_type BERT --project_name "BERTProject" --dataset_path "/path/to/dataset" --eval_only=True --snapme_test=True
```

#### Qformer Evaluation
```bash
python run.py --model_type Qformer --project_name "QformerProject" --dataset_path "/path/to/dataset" --eval_only=True --snapme_test=True
```

#### T5 Evaluation
```bash
python run.py --model_type T5 --project_name "T5Project" --dataset_path "/path/to/dataset" --eval_only=True --snapme_test=True
```


