import os
from typing import Callable, Optional, Tuple, Union
import torch

from transformers import LlavaForConditionalGeneration

class FreezeLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    def __init__(self, config, freeze_vision=True, freeze_llm=True): # vision_tower, multi_modal_projector, language_model
        super().__init__(config)
        # TODO from_pretrained 이후에 freeze 하는 방법..
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm

        ignore_prefix = set()
        if self.freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            ignore_prefix.add('vision_tower')

        if self.freeze_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
            ignore_prefix.add('language_model')

        self.set_ignore_keys(tuple(ignore_prefix))
    
    def set_ignore_keys(self, ignore_prefix=('vision_tower', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        self._keys_to_ignore_on_save = set(ignore_keys)
    
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