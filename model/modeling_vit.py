from typing import Union, Optional, Callable
import logging
import os

from transformers import ViTForImageClassification, ViTModel
from transformers.modeling_outputs import (
    ImageClassifierOutput,
)

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch

class MyViTForImageClassification(ViTForImageClassification): # TODO 완전 다시..
    def __init__ (self, config, model_name, vit_freeze=True, num_labels=1488, multi_classification=True):
        # TODO
        super().__init__(config)
        self.num_labels = num_labels
        self.vit_freeze = vit_freeze

        vit = ViTModel.from_pretrained(model_name)
        vit.pooler = None
        self.classifier = nn.Linear(vit.config.hidden_size, self.num_labels)

        self.vit = vit

        self.post_init()
        self.backbone_freeze()
        if self.vit_freeze:
            self.set_ignore_keys()
    
    def backbone_freeze(self):
        if self.vit_freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        num_train_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f'* Number of training paramters: {num_train_param}')
    

    def set_ignore_keys(self, ignore_prefix=('vit')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys for ViT backbone param.
        """
        
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)] # save only classifier head

        self._keys_to_ignore_on_save = set(ignore_keys)
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save the model using the traditional PyTorch way (pickle) by safe_serialization to False
        """
        super().save_pretrained(
            save_directory=save_directory, 
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )
    
    def save_pretrained_final(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save entire model (including ViT, LLM)
        """
        self._keys_to_ignore_on_save = None

        super().save_pretrained(
            save_directory=save_directory, 
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )
