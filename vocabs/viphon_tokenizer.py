import torch

from configs.viphon_bert_config import ViPhonBertConfig
from .Vietnamese_utils import analyse_Vietnamese

import re
from typing import *
import random
from collections import OrderedDict
import unicodedata

class VietnameseEncodedTokens(OrderedDict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"{key} not found")

    def get_fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.

        """
        return list(self.keys())

class ViPhonTokenizer:
    def __init__(self, config: ViPhonBertConfig):
        self.config = config
        self.analyze = analyse_Vietnamese

    def create_attention_mask(self, ids: torch.Tensor):
        ids = ids[:, 0]
        mask = (ids != self.config.pad_token_id).long()

        return mask
    
    def create_labels(self, input_ids: torch.Tensor):
        labels = torch.zeros_like(input_ids).fill_(-100).long()
        length, _ = input_ids.shape
        for idx in range(length-1):
            token_id = input_ids[idx+1, 0]
            if token_id in self.config.special_ids:
                continue
            if random.random() <= 0.15:
                    labels[idx+1, :] = input_ids[idx+1, :]
                    input_ids[idx+1, :] = self.config.mask_token_id

        return input_ids, labels
    
    def normalize(self, text: str):
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = re.sub(r"\s+", " ", text)
        special_tokens = [
            "0", "1", "2", "3", "4", "5", "6", 
            "7", "8", "9", "!", "@", "#", "$", 
            "%", "^", "&", "*", "(", ")", "'",
            "\"", "-", "=", "[", "]", "{", "}",
            "|", "\\", ":", ";", "<", ">", "/",
            "?", ".", ",", "_", "。", "·"
        ]
        pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        # Insert spaces around matched tokens
        text = re.sub(pattern, r" \1 ", text)
        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def encode(self, sentence: str) -> torch.Tensor:
        sentence = self.normalize(sentence)
        syllables = [
            (self.config.cls_token_id, ) * 3
        ]
        for word in sentence.split():
            components = self.analyze(word)
            if components:
                initial, rhyme, tone = components
                if rhyme in self.config.label2id:
                    syllables.append((
                        self.config.label2id[initial] if initial else self.config.empty_token_id,
                        self.config.label2id[rhyme],
                        self.config.label2id[tone] if tone else self.config.empty_token_id 
                    ))
                else:
                    syllables.append((self.config.unk_token_id, ) * 3)
            else:
                for char in word:
                    syllables.append(
                        (self.config.label2id[char], ) * 3 if char in self.config.label2id else (self.config.unk_token_id, ) * 3
                    )

        vec = torch.tensor(syllables).long()
        # truncate the input
        vec = vec[:self.config.max_length]

        return vec
    
    def __call__(self, sentence: str) -> VietnameseEncodedTokens:
        sentence_ids = self.encode(sentence)
        _, labels = self.create_labels(sentence_ids)
        return VietnameseEncodedTokens(
            input_ids = sentence_ids,
            labels = labels
        )


