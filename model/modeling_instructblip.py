import os
from typing import Callable, Optional, Tuple, Union
import warnings
# import logging
from transformers.utils import logging

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    BertForSequenceClassification,
    CLIPModel,
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
    T5Model
)
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput
from transformers import InstructBlipVisionModel, InstructBlipQFormerModel, AutoModel, AutoConfig
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.instructblip.modeling_instructblip import InstructBlipForConditionalGenerationModelOutput, InstructBlipPreTrainedModel
from transformers.models.auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.models.t5.modeling_t5 import T5ClassificationHead
from transformers.models.t5.configuration_t5 import T5Config
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from .modeling_t5 import QT5ForConditionalGeneration, T5ForClassification, ClassificationHead
from .modeling_output import QT5ClassifierOutput

logger = logging.get_logger(__name__)

class QformerInstructBlip(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = 'pixel_values'

    def __init__(self, config):
        super().__init__(config)

        self.vision_model = InstructBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # self.classification_input_size = config.num_query_tokens * config.qformer_config.hidden_size
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        language_model = QT5ForConditionalGeneration(config.text_config)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        self.post_init()
    
    # TODO combine with init
    def remove_llm(self, num_labels=1488, multi_classification=False, classifier_dropout=None):
        self.num_labels = num_labels
        self.multi_classification = multi_classification

        self.loss_fct = BCEWithLogitsLoss() if self.multi_classification else CrossEntropyLoss()

        self.language_model = None
        self.language_projection = None

        # TODO
        self.classification_input_size = self.config.num_query_tokens * self.config.qformer_config.hidden_size
        self.classification_head1 = nn.Linear(self.classification_input_size, self.config.qformer_config.hidden_size)
        self.classification_head2 = nn.Linear(self.config.qformer_config.hidden_size, self.num_labels)
        self.final_dropout = nn.Dropout(self.config.text_config.dropout_rate if classifier_dropout is None else classifier_dropout) # 0.1
        
        self.post_init()
        
        self.backbone_freeze()
    
    def backbone_freeze(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        self.set_ignore_keys()

        num_train_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'* Number of training paramters: {num_train_param}')
    
    def set_ignore_keys(self):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """

        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        # input_ids: Optional[torch.FloatTensor] = None,
        # attention_mask: Optional[torch.LongTensor] = None,
        # decoder_input_ids: Optional[torch.LongTensor] = None,
        # decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0] # (batch, 257, 1408)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        bs = image_embeds.shape[0]

        outputs = self.classification_head1(query_output.view(bs, -1))
        outputs = self.final_dropout(outputs)
        logits = self.classification_head2(outputs) # num_labels

        # logits = self.classification_head2(torch.mean(query_output, dim=1)) # mean pool
        
        loss = self.loss_fct(logits, labels.float()) ## TODO

        return QT5ClassifierOutput(
            loss=loss,
            logits=logits,
            # f1 = f1,
        )

    def get_metrics(self, logits, labels):
        if self.multi_classification: # f1
            preds = (logits.sigmoid() > 0.5).int() 
            # 'micro', 'macro',  'weighted'
            # TODO debug
            f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro') 
            return {'f1': f1}
                
        else: # accuracy
            preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            preds_one_hot = np.eye(self.num_labels)[preds]
            accuracy = accuracy_score(labels.detach().cpu().numpy(), preds_one_hot)
            return {'accuracy': accuracy}

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


class T5DecoderInstructBlipForClassification(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        self.language_hidden_size = config.text_config.hidden_size
        # vision_model, qformer, language_projection, language_model
        language_model = T5ForClassification(config.text_config)
        
        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        self.post_init()

        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False
    
    def to_classifier(self, num_labels, multi_classification):
        self.num_labels = num_labels
        self.multi_classification = multi_classification
        self.loss_fct = BCEWithLogitsLoss() if self.multi_classification else CrossEntropyLoss()

        self.classification_head = nn.Linear(self.language_hidden_size, self.num_labels)
        self.post_init()

        self.set_ignore_keys()

    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.
        """

        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained

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

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        ingredient_ids_one_hot,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        # if self.config.use_decoder_only_language_model:
        #     outputs = self.language_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #     )
        #     logits = outputs.logits if return_dict else outputs[0]
        #     loss = None
        #     # we compute the loss here since we need to take into account the sequence length of the query embeds
        #     if labels is not None:
        #         labels = labels.to(logits.device)
        #         logits = logits[:, -labels.size(1) :, :]
        #         # Shift so that tokens < n predict n
        #         shift_logits = logits[..., :-1, :].contiguous()
        #         shift_labels = labels[..., 1:].contiguous().to(logits.device)

        #         # Flatten the tokens
        #         loss_fct = CrossEntropyLoss(reduction="mean")

        #         loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        # else:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, # True? or override?
            return_dict=return_dict,
            labels=labels,
            decoder_last_hidden_only = True,
        )
        # loss = outputs.loss if return_dict else outputs[0]
        # logits = outputs.logits if return_dict else outputs[1]
        last_hidden = outputs['last_hidden_state'] # (batch_size, seq_len, hidden_size)
        last_token_mask = labels.eq(self.config.text_config.eos_token_id).to(last_hidden.device) # (batch_size, 128)
        batch_size = last_hidden.shape[0]
        assert last_token_mask.sum() == batch_size

        last_token_idx = last_token_mask.nonzero(as_tuple=True)[1]
        batch_range = torch.arange(batch_size, device=last_hidden.device)
        last_token_representation = last_hidden[batch_range, last_token_idx, :] # (batch_size, hidden_size)

        logits = self.classification_head(last_token_representation)

        loss = self.loss_fct(logits, ingredient_ids_one_hot)

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        # return InstructBlipForConditionalGenerationModelOutput(
        #     loss=loss,
        #     logits=logits,
        #     vision_outputs=vision_outputs,
        #     qformer_outputs=query_outputs,
        #     language_model_outputs=outputs,
        # )
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
            
    # def generate_and_logits(self):
    #     pass

class T5EncoderInstructBlipForSequenceClassification(InstructBlipForConditionalGeneration):

    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        self.language_hidden_size = config.text_config.hidden_size
    
    def to_classifier(self, num_labels, multi_classification):
        """
        Add classification head on top of language model encoder
        Remove language model decoder and lm_head
        """
        self.num_labels = num_labels
        self.multi_classification = multi_classification

        self.loss_fct = BCEWithLogitsLoss() if multi_classification else CrossEntropyLoss()

        t5config = T5Config()
        t5config.d_model = self.language_hidden_size
        t5config.num_labels = self.num_labels
        
        self.classification_head = T5ClassificationHead(t5config) ## 2 layer! (dense -> dropout -> out_proj)
        
        self.post_init()
        # remove decoder, lm_head
        self.language_model.decoder = None
        self.language_model.lm_head = None

        self.backbone_freeze()
    
    def backbone_freeze(self):
        
        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False

        # train query_token, qformer, language_projection, classification_head
        self.set_ignore_keys()

    def set_ignore_keys(self):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.
        """
        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained
    
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
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs.last_hidden_state
            batch_size = sequence_output.shape[0]

            eos_mask = input_ids.eq(self.config.text_config.eos_token_id).to(sequence_output.device)
            assert eos_mask.sum() == batch_size, 'Each input should have only one eos token'

            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")

            eos_idx = eos_mask.nonzero(as_tuple=True)[1]
            batch_range = torch.arange(batch_size, device=sequence_output.device)
            sentence_representation = sequence_output[batch_range, eos_idx, :] # (batch_size, hidden_size)
            # sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
            logits = self.classification_head(sentence_representation)

            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
class BERTInstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, bert_name, train_llm=False, train_vit=False, num_labels=1488):
        config = InstructBlipConfig()
        config.use_decoder_only_language_model = False
        
        super().__init__(config)
        
        self.num_labels = num_labels

        self.vision_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").vision_model

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)
        # self.qformer = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").qformer # pretrained qformer

        language_model = BertForSequenceClassification.from_pretrained(bert_name, problem_type="multi_label_classification", num_labels=self.num_labels)
        
        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, self.language_model.config.hidden_size)

        self.post_init()
        self.backbone_freeze(train_llm, train_vit)
    
    def backbone_freeze(self, train_llm=False, train_vit=False):

        if not train_vit:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if not train_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # train classifier
        for param in self.language_model.classifier.parameters():
            param.requires_grad = True

        self.set_ignore_keys(train_llm, train_vit)

    def set_ignore_keys(self, train_llm, train_vit, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        if train_llm:
            ignore_prefix = list(ignore_prefix)
            ignore_prefix.remove('language_model')
            ignore_prefix = tuple(ignore_prefix)
        
        if train_vit:
            ignore_prefix = list(ignore_prefix)
            ignore_prefix.remove('vision_model')
            ignore_prefix = tuple(ignore_prefix)
        
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        
        # include classifier to trainable param
        if not train_llm:
            ignore_keys.remove('language_model.classifier.weight')
            ignore_keys.remove('language_model.classifier.bias')

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

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels.float(), ## TODO
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

        # return InstructBlipForConditionalGenerationModelOutput(
        #     loss=loss,
        #     logits=logits,
        #     vision_outputs=vision_outputs,
        #     qformer_outputs=query_outputs,
        #     language_model_outputs=outputs,
        # )

class FreezeInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        # vision_model, qformer, language_projection, language_model
        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False

        self.set_ignore_keys()

    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.
        """

        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained


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

class QT5InstructBlipForClassification(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = 'pixel_values'

    def __init__(self, config):
        super().__init__(config)

        self.vision_model = InstructBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        language_model = QT5ForConditionalGeneration(config.text_config)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        self.final_dropout = nn.Dropout(config.text_config.dropout_rate)

        self.post_init()
    
    # TODO combine with init
    def learnable_query_init(self, num_query=8, num_labels=1488, freeze_qformer=False, multi_classification=False, classifier_dropout=None):
        self.num_query = num_query
        self.num_labels = num_labels
        self.freeze_qformer = freeze_qformer
        self.multi_classification = multi_classification

        self.loss_fct = BCEWithLogitsLoss() if self.multi_classification else CrossEntropyLoss()

        language_hidden = self.language_model.config.hidden_size

        ###
        # assert self.language_model.config.hidden_size % num_query == 0
        # self.reduction_size = self.language_model.config.hidden_size // self.num_query
        
        # self.reduction_layer = nn.Linear(self.language_model.config.hidden_size, self.reduction_size)




        # TODO reduction layer 살리기.. T5classificationhead 2 layers + dropout
        # combine_layer_size = language_hidden * self.num_query
        # self.combine_layer = nn.Linear(combine_layer_size, language_hidden)

        # self.classification_head = nn.Linear(language_hidden, self.num_labels)



        
        self.classification_head = ClassificationHead(num_query=self.num_query, hidden_size=language_hidden, num_labels=self.num_labels, dropout=self.config.text_config.dropout_rate if classifier_dropout is None else classifier_dropout) # text_config.dropout_rate: 0.1, text_config.classifier_dropout: 0.0
        
        self.decoder_query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query, language_hidden)
        )
        self.decoder_query_tokens.data.normal_(mean=0.0, std=0.02) # TODO std
        
        self.post_init() ## to init reduction_layer, classification_head 
        
        self.backbone_freeze()
    
    def backbone_freeze(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False
        
        if self.freeze_qformer:
            for param in self.qformer.parameters():
                param.requires_grad = False
        
        self.set_ignore_keys()

        num_train_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'* Number of training paramters: {num_train_param}')
    
    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        # all_keys = self.state_dict().keys()
        # ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        # self._keys_to_ignore_on_save = set(ignore_keys)

        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0] # (batch, 257, 1408)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        ## HERE: learnable query ##
        bs = input_ids.shape[0]
        decoder_query_tokens = self.decoder_query_tokens.expand(bs, -1, -1) # (batch_size, num_query, 2048)
        decoder_query_atts = torch.ones(decoder_query_tokens.size()[:-1], dtype=torch.long).to(input_ids.device) # (batch_size, num_query) # 0 when padding?
        ##

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_inputs_embeds=decoder_query_tokens,
            decoder_attention_mask=decoder_query_atts,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            labels=None,
        )

        last_hidden_state = outputs.decoder_hidden_states[-1] # (batch_size, num_query, 2048)

        # Concat queries -> combine layer -> dropout -> classification
        concat_last_hidden_state = last_hidden_state.view(bs, -1) # (batch_size, num_query * 2048)
        logits = self.classification_head(concat_last_hidden_state) # (batch_size, num_label)

        
        # combine_query_output = self.combine_layer(concat_last_hidden_state) # (batch_size, 2048)
        # combine_query_output = self.final_dropout(combine_query_output)
        # logits = self.classification_head(combine_query_output)

        
        # if self.num_query == 1:
        #     logits.squeeze_(1) ## in place
        loss = self.loss_fct(logits, labels.float()) ## TODO

        # pred_labels = (torch.sigmoid(logits.detach().cpu()) > 0.5).float()
        # f1 = f1_score(labels.detach().cpu(), pred_labels, average='micro')

        # output_dict = {
        #     'loss': loss,
        #     'logits': logits,
        # }
        # output_dict.update(self.get_metrics(logits, labels))

        # return output_dict # TODO training accuracy output

        return QT5ClassifierOutput(
            loss=loss,
            logits=logits,
            # f1 = f1,
        )

    def get_metrics(self, logits, labels):
        if self.multi_classification: # f1
            preds = (logits.sigmoid() > 0.5).int() 
            # 'micro', 'macro',  'weighted'
            # TODO debug
            f1 = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro') 
            return {'f1': f1}
                
        else: # accuracy
            preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
            preds_one_hot = np.eye(self.num_labels)[preds]
            accuracy = accuracy_score(labels.detach().cpu().numpy(), preds_one_hot)
            return {'accuracy': accuracy}

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

class CLIP_QT5InstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig, clip_model='openai/clip-vit-large-patch14-336', num_query=8, num_labels=1488):
        config.vision_config = clip_model # TODO config.qformer_config check. maybe vision->qformer project matrix 안 맞을듯
        config.use_decoder_only_language_model = False
        super().__init__(config)

        # self.vision_model = InstructBlipVisionModel(config.vision_config)
        # self.vision_model = CLIPVisionModel.from_pretrained(clip_model)
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_config) # hidden_size 1024
        config.qformer_config.encoder_hidden_size = self.vision_model.config.hidden_size # 1024
        ##
        # config.qformer_config.hidden_size = self.vision_model.config.hidden_size
        ##

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)) # 768
        self.qformer = InstructBlipQFormerModel(config.qformer_config)

        # self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)

        # if config.use_decoder_only_language_model:
        #     language_model = AutoModelForCausalLM.from_config(config.text_config)
        # else:
        #     language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        
        language_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").language_model
        # self.vision_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").vision_model

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, self.language_model.config.hidden_size)

        ## learnable query
        self.num_labels = num_labels
        self.num_query = num_query
        assert self.language_model.config.hidden_size % num_query == 0

        self.reduction_size = self.language_model.config.hidden_size // self.num_query ## TODO num_query 1일 때는 redection layer 안 하는 게 나은가
        self.reduction_layer = nn.Linear(self.language_model.config.hidden_size, self.reduction_size)
        self.classification_head = nn.Linear(self.language_model.config.hidden_size, self.num_labels)
        self.loss_fct = BCEWithLogitsLoss()
        self.decoder_query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query, self.language_model.config.hidden_size)
        )
        self.decoder_query_tokens.data.normal_(mean=0.0, std=0.02) # TODO std
        ##

        # Initialize weights and apply final processing
        self.post_init()

        self.backbone_freeze()
    
    def backbone_freeze(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False
        
        self.set_ignore_keys()

        num_train_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'* Number of training paramters: {num_train_param}')
    
    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        # all_keys = self.state_dict().keys()
        # ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        # self._keys_to_ignore_on_save = set(ignore_keys)

        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0] # last_hidden_state # (bs, 577, 1024)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        ## HERE ##
        bs = input_ids.shape[0]
        decoder_query_tokens = self.decoder_query_tokens.expand(bs, -1, -1) # (batch_size, num_query, 2048)
        decoder_query_atts = torch.ones(decoder_query_tokens.size()[:-1], dtype=torch.long).to(input_ids.device) # (batch_size, num_query)
        ##

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_inputs_embeds=decoder_query_tokens,
            decoder_attention_mask=decoder_query_atts,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            labels=None,
        )

        last_hidden_state = outputs.decoder_hidden_states[-1] # (batch_size, num_query, 2048)

        reduced_query = self.reduction_layer(last_hidden_state) # (batch_size, num_query, 2048%num_query) # 256
        reduced_query = reduced_query.view(bs, -1) # (batch_size, 2048) # concate

        logits = self.classification_head(reduced_query) # (batch_size, num_label)
        if self.num_query == 1:
            logits.squeeze_(1) ## in place
        loss = self.loss_fct(logits, labels)

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    
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


class temp_CLIP_QT5InstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, config):
        # config = InstructBlipConfig(model_name)
        super().__init__(config)
        # TODO - from_pretrained에서 어떻게 불러오고 이걸 또 어떻게 init 하는지...
        self.num_labels = 1488
        self.qformer_num_query = config.num_query_tokens
        
    # TODO better way to reinit
    def reinit(self, clip_model='openai/clip-vit-large-patch14-336', num_query=8, num_labels=1488):
        ## TODO reinit Q-former - mapping linear 사이즈 다시 정의?
        self.num_labels = num_labels
        self.num_query = num_query
        assert self.language_model.config.hidden_size % num_query == 0
        # TODO if num_query != 1: self.reduction_layer ~~  # if num_query is 1, no reduction layer needed
        self.reduction_size = self.language_model.config.hidden_size // self.num_query ## TODO ??? # 
        
        self.reduction_layer = nn.Linear(self.language_model.config.hidden_size, self.reduction_size)
        self.classification_head = nn.Linear(self.language_model.config.hidden_size, self.num_labels)
        self.loss_fct = BCEWithLogitsLoss()
        self.decoder_query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query, self.language_model.config.hidden_size)
        )
        self.decoder_query_tokens.data.normal_(mean=0.0, std=0.02) # TODO std

        ##
        # self.vision_model = CLIPModel.from_pretrained(clip_model)
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model)

        # project image features 1024 -> 768 (qformer hidden size)
        # self.vision_projection = nn.Linear(self.vision_model.config.hidden_size, self.qformer.config.hidden_size)
        
        self.backbone_freeze()
    
    def backbone_freeze(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False
        
        self.set_ignore_keys()

        num_train_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'* Number of training paramters: {num_train_param}')
    
    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        # all_keys = self.state_dict().keys()
        # ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        # self._keys_to_ignore_on_save = set(ignore_keys)

        trainable_keys = [name for name, param in self.named_parameters() if param.requires_grad]
        no_trained = self.state_dict().keys() - set(trainable_keys) # not trainable params
        self._keys_to_ignore_on_save = no_trained
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0] # last_hidden_state # (bs, 577, 1024)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        ## HERE ##
        bs = input_ids.shape[0]
        decoder_query_tokens = self.decoder_query_tokens.expand(bs, -1, -1) # (batch_size, num_query, 2048)
        decoder_query_atts = torch.ones(decoder_query_tokens.size()[:-1], dtype=torch.long).to(input_ids.device) # (batch_size, num_query)
        ##

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_inputs_embeds=decoder_query_tokens,
            decoder_attention_mask=decoder_query_atts,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            labels=None,
        )

        last_hidden_state = outputs.decoder_hidden_states[-1] # (batch_size, num_query, 2048)

        reduced_query = self.reduction_layer(last_hidden_state) # (batch_size, num_query, 2048%num_query) # 256
        reduced_query = reduced_query.view(bs, -1) # (batch_size, 2048) # concate

        logits = self.classification_head(reduced_query) # (batch_size, num_label)
        if self.num_query == 1:
            logits.squeeze_(1) ## in place
        loss = self.loss_fct(logits, labels)

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    
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

class CLIP_BERTInstructBlipForConditionalGeneration(InstructBlipPreTrainedModel):
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, config: InstructBlipConfig, clip_model='openai/clip-vit-large-patch14-336', bert_name='bert-large-uncased', num_labels=1488):
        config.vision_config = clip_model 
        config.use_decoder_only_language_model = False
        
        super().__init__(config)

        self.num_labels = num_labels
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_config) # hidden_size 1024
        config.qformer_config.encoder_hidden_size = self.vision_model.config.hidden_size # 1024

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)) # 768
        self.qformer = InstructBlipQFormerModel(config.qformer_config) # __init__ call 될 때마다 다르게 init

        language_model = BertForSequenceClassification.from_pretrained(bert_name, problem_type="multi_label_classification", num_labels=self.num_labels)
        # bert-base-uncased: 110M

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, language_model.config.hidden_size)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        # Initialize weights and apply final processing
        # self.post_init(): additional initialization steps that are not covered by the constructors of the individual components.
        # self.vision_mode, self.qformer, self.language_model, self.query_tokens are already properly initialized through pre-trained weights or their own constructors -> only self.language_projection is being initialized
        self.post_init()

        self.backbone_freeze()
    
    def backbone_freeze(self, train_llm=False, train_vit=False):

        if not train_vit:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if not train_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # train classifier
        for param in self.language_model.classifier.parameters():
            param.requires_grad = True

        self.set_ignore_keys(train_llm, train_vit)

    def set_ignore_keys(self, train_llm, train_vit, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        if train_llm:
            ignore_prefix = list(ignore_prefix)
            ignore_prefix.remove('language_model')
            ignore_prefix = tuple(ignore_prefix)
        
        if train_vit:
            ignore_prefix = list(ignore_prefix)
            ignore_prefix.remove('vision_model')
            ignore_prefix = tuple(ignore_prefix)
        
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        
        # include classifier to trainable param
        if not train_llm:
            ignore_keys.remove('language_model.classifier.weight')
            ignore_keys.remove('language_model.classifier.bias')

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

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

# TODO class CLIP_BERTInstructBlipForConditionalGeneration
# TODO maybe model renaming... no longer conditional generation