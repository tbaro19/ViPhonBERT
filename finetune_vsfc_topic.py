import argparse
import json
import logging
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel
from safetensors.torch import load_file

from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# THIẾT LẬP NHÃN CHO UIT-VSFC (TOPIC)
# ==========================================
LABEL_MAP = {
    "lecturer": 0,
    "training_program": 1,
    "facility": 2,
    "others": 3
}
NUM_LABELS = len(LABEL_MAP)

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"👉 Đã thiết lập mã Seed cố định: {seed}")


class ViPhonBertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__() 
        self.num_labels = num_labels
        self.hidden_size = config.hidden_size
        
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, labels=None):
        onset_ids = input_ids[:, :, 0]
        rhyme_ids = input_ids[:, :, 1]
        tone_ids = input_ids[:, :, 2]

        onset_emb = self.shared_embeddings(onset_ids)
        rhyme_emb = self.shared_embeddings(rhyme_ids)
        tone_emb = self.shared_embeddings(tone_ids)

        pinyin_emb = torch.cat([onset_emb, rhyme_emb, tone_emb], dim=-1)
        inputs_embeds = self.fc_emb(pinyin_emb)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        sequence_output = outputs.last_hidden_state
        
        # Mean Pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled_output = sum_embeddings / sum_mask
        
        cls_output = self.dropout(mean_pooled_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits

def load_vsfc_topic_data(data_path):
    examples = []
    if not os.path.exists(data_path):
        logger.warning(f"Tệp dữ liệu không tồn tại: {data_path}")
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        # VSFC lưu dưới dạng List, không phải Dict
        data = json.load(f) 
        for i, val in enumerate(data):
            sentence = val.get("sentence", "").strip()
            topic = val.get("topic", "").strip()
            
            if not sentence or not topic:
                continue
                
            if topic not in LABEL_MAP:
                logger.warning(f"Nhãn topic lạ '{topic}' tại index {i}. Bỏ qua.")
                continue
                
            examples.append({
                "text": sentence,
                "label": LABEL_MAP[topic]
            })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    pad_token = (tokenizer.config.pad_token_id,) * 3
    features = []
    for example in tqdm(examples, desc="Converting features"):
        input_ids = tokenizer.encode(example["text"]).tolist()[:max_seq_length]
        attention_mask = [1] * len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": example["label"]
        })
    return features

def create_dataloader(features, batch_size, is_training=True):
    if not features: return None
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f["label_id"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

def evaluate(model, dataloader, device):
    if dataloader is None: return 0, 0, 0
    model.eval()
    eval_loss, nb_eval_steps = 0, 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        eval_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=-1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
    macro_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0
    return eval_loss, acc, macro_f1

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.init_checkpoint, "config.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ViPhonBertConfig(**config_dict)
    tokenizer = ViPhonTokenizer(config) 
    model = ViPhonBertForSequenceClassification(config, num_labels=NUM_LABELS).to(device)
    
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.init_checkpoint, "pytorch_model.bin")
    
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path) if weights_path.endswith('.safetensors') else torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    cache_train = os.path.join(args.data_dir, f"cached_vsfc_topic_train_{args.max_seq_length}.pt")
    cache_dev = os.path.join(args.data_dir, f"cached_vsfc_topic_dev_{args.max_seq_length}.pt")
    cache_test = os.path.join(args.data_dir, f"cached_vsfc_topic_test_{args.max_seq_length}.pt")

    if os.path.exists(cache_train): train_features = torch.load(cache_train)
    else:
        train_features = convert_examples_to_features(load_vsfc_topic_data(os.path.join(args.data_dir, "UIT-VSFC-train.json")), tokenizer, args.max_seq_length)
        torch.save(train_features, cache_train)

    if os.path.exists(cache_dev): dev_features = torch.load(cache_dev)
    else:
        dev_features = convert_examples_to_features(load_vsfc_topic_data(os.path.join(args.data_dir, "UIT-VSFC-dev.json")), tokenizer, args.max_seq_length)
        torch.save(dev_features, cache_dev)

    if os.path.exists(cache_test): test_features = torch.load(cache_test)
    else:
        test_features = convert_examples_to_features(load_vsfc_topic_data(os.path.join(args.data_dir, "UIT-VSFC-test.json")), tokenizer, args.max_seq_length)
        if test_features: torch.save(test_features, cache_test)

    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)
    test_dataloader = create_dataloader(test_features, args.eval_batch_size, is_training=False) if test_features else None

    t_total = len(train_dataloader) * args.epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda') 

    best_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model_vsfc_topic.pt")

    for epoch in range(args.epochs):
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                with autocast('cuda'):
                    loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        eval_loss, eval_acc, eval_f1 = evaluate(model, dev_dataloader, device)
        logger.info(f"Epoch {epoch+1} - Loss: {eval_loss:.4f} - Acc: {eval_acc*100:.2f}% - Macro F1: {eval_f1*100:.2f}%")

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✨ LƯU KỶ LỤC TỐI ƯU MỚI: Macro F1 = {best_f1*100:.2f}% tại {best_model_path}")

    logger.info("***** Bắt đầu chấm điểm trên tập Test.json *****")
    if test_dataloader is not None and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc, test_f1 = evaluate(model, test_dataloader, device)
        logger.info(f"🎯 KẾT QUẢ TEST TOPIC - Loss: {test_loss:.4f} | Acc: {test_acc*100:.2f}% | Macro F1: {test_f1*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./vsfc_topic_outputs")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()