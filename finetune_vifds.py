import argparse
import json
import logging
import os
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

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
# THIẾT LẬP NHÃN CHO UIT-ViFDs (ABSA Task)
# ==========================================
# 10 Khía cạnh (Aspects) có trong bộ dữ liệu
ASPECTS = [
    "SCREEN", "CAMERA", "FEATURES", "BATTERY", 
    "PERFORMANCE", "DESIGN", "PRICE", "GENERAL", 
    "SER&ACC", "OTHERS"
]
ASPECT_MAP = {aspect: idx for idx, aspect in enumerate(ASPECTS)}
NUM_ASPECTS = len(ASPECTS)

# 5 Trạng thái cảm xúc cho mỗi khía cạnh
SENTIMENT_MAP = {
    "None": 0,      # Không được nhắc đến trong câu
    "Positive": 1,  # Tích cực
    "Negative": 2,  # Tiêu cực
    "Neutral": 3,   # Trung lập
    None: 4,        # Có nhắc đến nhưng không rõ cảm xúc (null)
    "null": 4
}
NUM_SENTIMENTS = len(SENTIMENT_MAP)

def set_seed(seed: int):
    """Cố định seed để đảm bảo tính tái lặp"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"👉 Đã thiết lập mã Seed cố định: {seed}")


# ==========================================
# 1. KIẾN TRÚC MÔ HÌNH (ASPECT-BASED SENTIMENT ANALYSIS)
# ==========================================
class ViPhonBertForABSA(nn.Module):
    """
    Kiến trúc ViPhonBERT dự đoán đồng thời cảm xúc của 10 khía cạnh.
    """
    def __init__(self, config, num_aspects, num_sentiments):
        super().__init__() 
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.hidden_size = config.hidden_size
        
        # Shared Embeddings
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        # BERT Encoder
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Lớp phân loại: Ánh xạ từ Hidden Size -> (10 aspects * 5 sentiments)
        self.classifier = nn.Linear(config.hidden_size, num_aspects * num_sentiments)

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
        cls_output = sequence_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Shape: (batch_size, num_aspects * num_sentiments)
        logits = self.classifier(cls_output)
        # Reshape: (batch_size, num_aspects, num_sentiments)
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)

        loss = None
        if labels is not None:
            # Labels shape: (batch_size, num_aspects)
            loss_fct = nn.CrossEntropyLoss()
            # Duỗi tensor để tính loss chung cho tất cả các khía cạnh
            loss = loss_fct(logits.view(-1, self.num_sentiments), labels.view(-1))

        return loss, logits


# ==========================================
# 2. XỬ LÝ DỮ LIỆU & CACHING
# ==========================================
def load_vifds_data(data_path):
    examples = []
    if not os.path.exists(data_path):
        logger.warning(f"Tệp dữ liệu không tồn tại: {data_path}")
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, val in data.items():
            if not val.get("comment") or not isinstance(val.get("label"), list):
                continue
                
            # Khởi tạo vector nhãn: 10 khía cạnh mặc định là 0 (None)
            aspect_labels = [0] * NUM_ASPECTS
            
            for item in val["label"]:
                aspect = item.get("aspect")
                sentiment = item.get("sentiment")
                
                if aspect in ASPECT_MAP:
                    asp_idx = ASPECT_MAP[aspect]
                    sent_idx = SENTIMENT_MAP.get(sentiment, 4) # 4 là Null
                    aspect_labels[asp_idx] = sent_idx
                    
            examples.append({
                "id": key,
                "text": val["comment"],
                "labels": aspect_labels
            })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    pad_token = (tokenizer.config.pad_token_id,) * 3
    features = []
    
    for example in tqdm(examples, desc="Converting features"):
        text = example["text"]
        labels = example["labels"] # List 10 phần tử
        
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
            "labels": labels
        })
        
    return features

def create_dataloader(features, batch_size, is_training=True):
    if not features:
        return None
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


# ==========================================
# 3. ĐÁNH GIÁ (EVALUATION)
# ==========================================
def evaluate(model, dataloader, device):
    if dataloader is None:
        return 0, 0, 0
    
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        eval_loss += loss.item()
        
        # Dự đoán ra class có xác suất cao nhất cho từng khía cạnh
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy() # (batch, 10)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 1. Aspect-level Accuracy (Độ chính xác trên từng khía cạnh độc lập)
    aspect_acc = np.sum(all_preds == all_labels) / (all_labels.shape[0] * NUM_ASPECTS)
    
    # 2. Strict Sentence Accuracy (Đúng tuyệt đối cả 10 khía cạnh trong câu)
    strict_acc = np.sum(np.all(all_preds == all_labels, axis=1)) / all_labels.shape[0]
    
    # 3. Macro F1 trên tất cả các dự đoán khía cạnh
    macro_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='macro', zero_division=0) if all_labels.size > 0 else 0
    
    return eval_loss, aspect_acc, strict_acc, macro_f1


# ==========================================
# 4. TRAINING LOOP VÀ CHẤM ĐIỂM TEST
# ==========================================
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading Config từ: {args.init_checkpoint}")
    config_path = os.path.join(args.init_checkpoint, "config.json")
    weights_path = os.path.join(args.init_checkpoint, "model.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(args.init_checkpoint, "pytorch_model.bin")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ViPhonBertConfig(**config_dict)
    tokenizer = ViPhonTokenizer(config) 
    
    model = ViPhonBertForABSA(
        config=config, 
        num_aspects=NUM_ASPECTS, 
        num_sentiments=NUM_SENTIMENTS
    ).to(device)
    
    logger.info(f"Loading Weights từ: {weights_path}")
    if os.path.exists(weights_path):
        if weights_path.endswith('.safetensors'):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=device)
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys (do thêm Classifier mới): {missing_keys}")
    else:
        logger.info("❌ Không tìm thấy checkpoint. Train từ Random Initialization.")

    # --- XỬ LÝ CACHE ---
    cache_train_path = os.path.join(args.data_dir, f"cached_vifds_train_{args.max_seq_length}.pt")
    cache_dev_path = os.path.join(args.data_dir, f"cached_vifds_dev_{args.max_seq_length}.pt")
    cache_test_path = os.path.join(args.data_dir, f"cached_vifds_test_{args.max_seq_length}.pt")

    # Tập Train
    if os.path.exists(cache_train_path):
        train_features = torch.load(cache_train_path)
    else:
        train_examples = load_vifds_data(os.path.join(args.data_dir, "train.json"))
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
        torch.save(train_features, cache_train_path)

    # Tập Dev
    if os.path.exists(cache_dev_path):
        dev_features = torch.load(cache_dev_path)
    else:
        dev_examples = load_vifds_data(os.path.join(args.data_dir, "dev.json"))
        dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
        torch.save(dev_features, cache_dev_path)

    # Tập Test
    if os.path.exists(cache_test_path):
        test_features = torch.load(cache_test_path)
    else:
        test_examples = load_vifds_data(os.path.join(args.data_dir, "test.json"))
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)
        if test_features:
            torch.save(test_features, cache_test_path)

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

    logger.info("***** Bắt đầu tiến trình Huấn luyện UIT-ViFDs (ABSA) *****")
    best_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model_vifds.pt")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()

                with autocast('cuda'):
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        logger.info(f"***** Đánh giá tập Dev (Epoch {epoch+1}) *****")
        eval_loss, aspect_acc, strict_acc, eval_f1 = evaluate(model, dev_dataloader, device)
        
        logger.info(f"Epoch {epoch+1} - Loss: {eval_loss:.4f} - Aspect Acc: {aspect_acc*100:.2f}% - Strict Sentence Acc: {strict_acc*100:.2f}% - Macro F1: {eval_f1*100:.2f}%")

        # Ta dùng Aspect-level Accuracy làm tiêu chí đánh giá mô hình tốt nhất
        if aspect_acc > best_acc:
            best_acc = aspect_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✨ LƯU KỶ LỤC TỐI ƯU MỚI: Aspect Acc = {best_acc*100:.2f}% tại {best_model_path}")

    # ---------------------------------------------
    # 4.2 ĐÁNH GIÁ TRÊN TẬP TEST SAU KHI TRAIN XONG
    # ---------------------------------------------
    logger.info("==================================================")
    logger.info("***** Bắt đầu chấm điểm trên tập Test.json *****")
    if test_dataloader is not None and os.path.exists(best_model_path):
        logger.info("Đang nạp lại trọng số tốt nhất từ quá trình huấn luyện...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        test_loss, t_aspect_acc, t_strict_acc, t_f1 = evaluate(model, test_dataloader, device)
        logger.info(f"🎯 KẾT QUẢ TEST CUỐI CÙNG - Loss: {test_loss:.4f}")
        logger.info(f"🎯 Aspect-level Accuracy: {t_aspect_acc*100:.2f}% (Chính xác khía cạnh độc lập)")
        logger.info(f"🎯 Strict Sentence Accuracy: {t_strict_acc*100:.2f}% (Đúng toàn bộ 10 khía cạnh trong 1 câu)")
        logger.info(f"🎯 Macro F1: {t_f1*100:.2f}%")
    else:
        logger.info("⚠️ Bỏ qua bước Test (không tìm thấy test.json hoặc model chưa được lưu).")
    logger.info("==================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa train.json, dev.json, test.json")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc")
    parser.add_argument("--output_dir", type=str, default="./vifds_outputs", help="Thư mục lưu output")
    
    parser.add_argument("--max_seq_length", type=int, default=256, help="Độ dài tối đa của câu bình luận")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size đánh giá/test")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()