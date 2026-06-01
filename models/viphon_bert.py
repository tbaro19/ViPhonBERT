import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.bert import BertModel

from configs.viphon_bert_config import ViPhonBertConfig

class ViPhonBert(PreTrainedModel):
    def __init__(self, config: ViPhonBertConfig):
        super().__init__(config)
        self.config = config
        
        # SHARED EMBEDDING CHO CẢ 3 THÀNH PHẦN
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(
            in_features=config.hidden_size*3,
            out_features=config.hidden_size
        )
        
        # 2. BERT ENCODER
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None # Bỏ word_embedding mặc định
        
        # 3. BA LỚP LINEAR DỰ ĐOÁN
        self.fc_onset = nn.Linear(config.hidden_size, config.vocab_size)
        self.fc_rhyme = nn.Linear(config.hidden_size, config.vocab_size)
        self.fc_tone = nn.Linear(config.hidden_size, config.vocab_size)
        
        # 4. HÀM LOSS CHUNG
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        bs, len, _ = input_ids.shape
        inputs_embeds = self.shared_embeddings(input_ids) # (bs, len, 3, dim)
        inputs_embeds = inputs_embeds.reshape(bs, len, -1) # (bs, len, 3*dim)
        inputs_embeds = self.fc_emb(inputs_embeds) # (bs, len, dim)

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # (B, L, 768)
        
        # Đi qua 3 FC Heads
        logit_onset = self.fc_onset(sequence_output) # (B, L, Vocab_size)
        logit_rhyme = self.fc_rhyme(sequence_output)
        logit_tone  = self.fc_tone(sequence_output)
        
        logits = torch.stack([logit_onset, logit_rhyme, logit_tone], dim=2) # (B, L, 3, Vocab_size)
        
        loss = None
        if labels is not None:
            _, _, _, V = logits.shape
            # Duỗi tensor để tính 1 hàm loss duy nhất 
            active_logits = logits.view(-1, V) # (B * L * 3, Vocab_size)
            active_labels = labels.view(-1)    # (B * L * 3)
            loss = self.loss_fn(active_logits, active_labels)
            
        return {"loss": loss, "logits": logits}
