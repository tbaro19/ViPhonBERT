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
# CẤU HÌNH NHÃN UIT-ViFDs
# ==========================================
ASPECTS = [
    "SCREEN", "CAMERA", "FEATURES", "BATTERY", 
    "PERFORMANCE", "DESIGN", "PRICE", "GENERAL", 
    "SER&ACC", "OTHERS"
]
ASPECT_MAP = {aspect: idx for idx, aspect in enumerate(ASPECTS)}
NUM_ASPECTS = len(ASPECTS)

SENTIMENT_MAP = {
    "None": 0, "Positive": 1, "Negative": 2, "Neutral": 3, "null": 4
}
NUM_SENTIMENTS = len(SENTIMENT_MAP)
INVERSE_SENTIMENT_MAP = {v: k for k, v in SENTIMENT_MAP.items()}

# ==========================================
# KIẾN TRÚC MÔ HÌNH (ABSA MODEL)
# ==========================================
class ViPhonBertForABSA(nn.Module):
    def __init__(self, config, num_aspects, num_sentiments):
        super().__init__() 
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
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
        self.classifier = nn.Linear(config.hidden_size, num_aspects * num_sentiments)

    def forward(self, input_ids, attention_mask=None, labels=None):
        onset_ids = input_ids[:, :, 0]
        rhyme_ids = input_ids[:, :, 1]
        tone_ids = input_ids[:, :, 2]

        onset_emb = self.shared_embeddings(onset_ids)
        rhyme_emb = self.shared_embeddings(rhyme_ids)
        tone_emb = self.shared_embeddings(tone_ids)

        pinyin_emb = torch.cat([onset_emb, rhyme_emb, tone_emb], dim=-1)
        inputs_embeds = self.fc_emb(pinyin_emb)

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        logits = self.classifier(cls_output)
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_sentiments), labels.view(-1))

        return loss, logits

# ==========================================
# XỬ LÝ DỮ LIỆU (Fallback nếu không có Cache)
# ==========================================
def load_vifds_data(data_path):
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, val in data.items():
            if not val.get("comment") or not isinstance(val.get("label"), list):
                continue
            aspect_labels = [0] * NUM_ASPECTS
            for item in val["label"]:
                aspect = item.get("aspect")
                sentiment = item.get("sentiment")
                if aspect in ASPECT_MAP:
                    asp_idx = ASPECT_MAP[aspect]
                    sent_idx = SENTIMENT_MAP.get(sentiment, 4)
                    aspect_labels[asp_idx] = sent_idx
                    
            examples.append({"id": key, "text": val["comment"], "labels": aspect_labels})
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    pad_token = (tokenizer.config.pad_token_id,) * 3
    features = []
    for example in tqdm(examples, desc="Converting Test features"):
        text = example["text"]
        labels = example["labels"]
        
        input_ids_tensor = tokenizer.encode(text) 
        input_ids = input_ids_tensor.tolist()
        input_ids = input_ids[:max_seq_length]
        
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length

        features.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
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
    
    all_preds = np.array(all_preds)   # (N, 10)
    all_labels = np.array(all_labels) # (N, 10)
    
    # Tính các metric tổng quan
    aspect_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    strict_acc = np.sum(np.all(all_preds == all_labels, axis=1)) / all_labels.shape[0]
    macro_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='macro', zero_division=0)
    
    # Phân tích báo cáo chi tiết cho từng Khía cạnh độc lập
    aspect_reports = []
    sentiment_names = [INVERSE_SENTIMENT_MAP[i] for i in range(NUM_SENTIMENTS)]
    
    for i, aspect_name in enumerate(ASPECTS):
        report_dict = classification_report(
            all_labels[:, i], 
            all_preds[:, i], 
            target_names=sentiment_names, 
            output_dict=True, 
            zero_division=0
        )
        aspect_f1 = report_dict['macro avg']['f1-score']
        aspect_reports.append(f"   * Aspect {aspect_name:<12} -> Macro F1: {aspect_f1*100:.2f}%")

    return eval_loss, aspect_acc, strict_acc, macro_f1, aspect_reports

def main():
    parser = argparse.ArgumentParser(description="Chấm điểm tập Test cho mô hình ViPhonBERT (ViFDs ABSA)")
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa test.json hoặc file cache")
    parser.add_argument("--config_dir", type=str, required=True, help="Thư mục chứa config.json gốc")
    parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn đến file checkpoint .pt tốt nhất")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Độ dài tối đa chuỗi")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size để inference")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Khởi tạo Config & Tokenizer
    logger.info("Loading Config & Tokenizer...")
    config_path = os.path.join(args.config_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ViPhonBertConfig(**config_dict)
    tokenizer = ViPhonTokenizer(config)
    
    # 2. Khởi tạo Model & Nạp trọng số tốt nhất
    logger.info(f"Đang nạp trọng số mô hình từ: {args.model_path}")
    model = ViPhonBertForABSA(config=config, num_aspects=NUM_ASPECTS, num_sentiments=NUM_SENTIMENTS)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file checkpoint tại: {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # 3. Chuẩn bị Dữ liệu Test
    cache_test_path = os.path.join(args.data_dir, f"cached_vifds_test_{args.max_seq_length}.pt")
    
    if os.path.exists(cache_test_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu Test, đang nạp từ: {cache_test_path}")
        test_features = torch.load(cache_test_path)
    else:
        logger.info("❌ Không tìm thấy file cache. Đang xử lý từ file test.json gốc...")
        test_json_path = os.path.join(args.data_dir, "test.json")
        if not os.path.exists(test_json_path):
            raise FileNotFoundError(f"Không tìm thấy file {test_json_path}")
            
        test_examples = load_vifds_data(test_json_path)
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)
    
    all_input_ids = torch.tensor([f["input_ids"] for f in test_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in test_features], dtype=torch.long)
    all_labels = torch.tensor([f["labels"] for f in test_features], dtype=torch.long)

    test_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    # 4. Đánh giá kết quả trên tập Test
    logger.info("==================================================")
    logger.info("***** BẮT ĐẦU CHẤM ĐIỂM TRÊN TẬP TEST (ABSA) *****")
    test_loss, aspect_acc, strict_acc, macro_f1, aspect_reports = evaluate(model, test_dataloader, device)
    
    logger.info(f"🎯 KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG:")
    logger.info(f"   - Test Loss:               {test_loss:.4f}")
    logger.info(f"   - Aspect-level Accuracy:   {aspect_acc*100:.2f}% (Độ chính xác tính trên các cell khía cạnh độc lập)")
    logger.info(f"   - Strict Sentence Acc:     {strict_acc*100:.2f}% (Đúng toàn bộ cả 10 khía cạnh trong cùng 1 câu)")
    logger.info(f"   - **Tổng quát Macro F1**:    {macro_f1*100:.2f}%")
    logger.info("\n📊 BÁO CÁO MACRO F1 CHI TIẾT TỪNG KHÍA CẠNH (ASPECTS):")
    for rep in aspect_reports:
        logger.info(rep)
    logger.info("==================================================")

if __name__ == "__main__":
    main()