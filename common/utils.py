import os
import torch
from transformers.utils import logging 

logger = logging.get_logger(__name__)

def load_eval_checkpoint(model, eval_checkpoint):
    if eval_checkpoint is None:
        raise ValueError("eval_check_point should be provided in eval_only mode")
    
    shard_files = [file for file in os.listdir(eval_checkpoint) if file.startswith('pytorch_model') and file.endswith('.bin')]
    shard_files.sort()

    checkpoint = {}
    for shard in shard_files:
        shard_path = os.path.join(eval_checkpoint, shard)
        state_dict_shard = torch.load(shard_path, map_location='cpu')
        checkpoint.update(state_dict_shard)

    model_state_dict = model.state_dict()

    model_state_dict.update(checkpoint)
    model.load_state_dict(model_state_dict)
    logger.info(f"Param loaded from checkpoint{eval_checkpoint}")

    return model