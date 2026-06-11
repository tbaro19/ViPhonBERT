import argparse
import json
import logging
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel

from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# THIẾT LẬP NHÃN
# ==========================================
LABEL_MAP = {
    "clean": 0,
    "offensive": 1,
    "hate": 2
}
NUM_LABELS = len(LABEL_MAP)
INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# KIẾN TRÚC MÔ HÌNH (SEQUENCE CLASSIFICATION)
# ==========================================
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
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return loss, logits

# ==========================================
# XỬ LÝ DỮ LIỆU (Fallback nếu thiếu Cache)
# ==========================================
def load_vihsd_data(data_path):
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, val in data.items():
            if not val.get("comment") or val.get("label") is None:
                continue
            label = val["label"]
            if label not in LABEL_MAP:
                continue
            examples.append({
                "id": key,
                "text": val["comment"],
                "label": LABEL_MAP[label]
            })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    pad_token = (tokenizer.config.pad_token_id,) * 3
    features = []
    
    for example in tqdm(examples, desc="Converting Test features"):
        text = example["text"]
        label_id = example["label"]
        
        input_ids_tensor = tokenizer.encode(text) 
        input_ids = input_ids_tensor.tolist()
        input_ids = input_ids[:max_seq_length]
        
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": label_id
        })
    return features

# ==========================================
# TIẾN TRÌNH ĐÁNH GIÁ (EVALUATION)
# ==========================================
def evaluate(model, dataloader, device):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating on Test Set"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        eval_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if len(all_labels) > 0 else 0
    
    # Tạo bảng báo cáo chi tiết từng nhãn
    target_names = [INVERSE_LABEL_MAP[i] for i in range(NUM_LABELS)]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    
    return eval_loss, accuracy, macro_f1, report

def main():
    parser = argparse.ArgumentParser(description="Chấm điểm tập Test cho mô hình ViPhonBERT (ViHSD)")
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa test.json hoặc file cache")
    parser.add_argument("--config_dir", type=str, required=True, help="Thư mục chứa config.json gốc")
    parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn đến file checkpoint .pt tốt nhất")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Độ dài tối đa chuỗi")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size để inference")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo Config & Tokenizer
    logger.info("Loading Config & Tokenizer...")
    config_path = os.path.join(args.config_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ViPhonBertConfig(**config_dict)
    tokenizer = ViPhonTokenizer(config)
    
    # 2. Khởi tạo Model & Nạp trọng số
    logger.info(f"Đang nạp trọng số mô hình từ: {args.model_path}")
    model = ViPhonBertForSequenceClassification(config, num_labels=NUM_LABELS)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file checkpoint tại: {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # 3. Chuẩn bị Dữ liệu Test
    cache_test_path = os.path.join(args.data_dir, f"cached_vihsd_test_{args.max_seq_length}.pt")
    
    if os.path.exists(cache_test_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu Test, đang nạp từ: {cache_test_path}")
        test_features = torch.load(cache_test_path)
    else:
        logger.info("❌ Không tìm thấy file cache. Đang xử lý từ file test.json gốc...")
        test_json_path = os.path.join(args.data_dir, "test.json")
        if not os.path.exists(test_json_path):
            raise FileNotFoundError(f"Không tìm thấy file {test_json_path}")
            
        test_examples = load_vihsd_data(test_json_path)
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)
    
    all_input_ids = torch.tensor([f["input_ids"] for f in test_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f["label_id"] for f in test_features], dtype=torch.long)

    test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # 4. Đánh giá kết quả
    logger.info("==================================================")
    logger.info("***** BẮT ĐẦU CHẤM ĐIỂM F1 MACRO TRÊN TẬP TEST *****")
    test_loss, test_acc, test_f1, detail_report = evaluate(model, test_dataloader, device)
    
    logger.info(f"🎯 KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG:")
    logger.info(f"   - Test Loss:       {test_loss:.4f}")
    logger.info(f"   - Test Accuracy:   {test_acc*100:.2f}%")
    logger.info(f"   - **Test F1 Macro**:   {test_f1*100:.2f}%")
    logger.info("\n📊 BÁO CÁO CHI TIẾT TỪNG NHÃN:\n" + detail_report)
    logger.info("==================================================")

if __name__ == "__main__":
    main()