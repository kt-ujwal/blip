from typing import List, Optional, Union

from transformers import InstructBlipProcessor, AutoTokenizer, CLIPProcessor, CLIPImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.file_utils import TensorType

class BERTInstructBlipProcessor(InstructBlipProcessor):
    # TODO
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor, tokenizer, qformer_tokenizer)
    
    def to_bert(self, bert_name):
        # BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name) # TODO

class CLIP_QT5InstructBlipProcessor(InstructBlipProcessor):
    
    # TODO
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor, tokenizer, qformer_tokenizer)
    
    def to_clip(self, clip_model = 'openai/clip-vit-large-patch14-336'):
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model)

class CLIP_BERTInstructBlipProcessor(InstructBlipProcessor):
    # TODO
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor, tokenizer, qformer_tokenizer)
    
    def to_clip_bert(self, clip_model = 'openai/clip-vit-large-patch14-336', bert_name='bert-large-uncased'):
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)


