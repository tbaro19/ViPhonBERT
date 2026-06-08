import argparse
import json
import logging
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, BertModel

from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# THIẾT LẬP DANH SÁCH NHÃN BIO CHO NIIVTB
# ==========================================
UNIQUE_POS = ["Nr", "Aa", "Nt", "Nn", "Vv", "PU", "Num", "Cs", "R", "Nc", "Nw", "Cp", "Pp", "M", "E", "Nu", "Nun", "Sv", "X", "VA", "ID", "FW"]
LABEL_MAP = {"O": 0}
for pos in UNIQUE_POS:
    LABEL_MAP[f"B-{pos}"] = len(LABEL_MAP)
    LABEL_MAP[f"I-{pos}"] = len(LABEL_MAP)

NUM_LABELS = len(LABEL_MAP)

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
# 1. KIẾN TRÚC MÔ HÌNH (MODEL WRAPPER)
# ==========================================
class ViPhonBertForTokenClassification(nn.Module):
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
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

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
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

        return loss, logits


# ==========================================
# 2. XỬ LÝ DỮ LIỆU & CACHING
# ==========================================
def load_niivtb_data(data_path):
    examples = []
    if not os.path.exists(data_path):
        logger.warning(f"Tệp dữ liệu không tồn tại: {data_path}")
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for key, val in data.items():
            if not val.get("text") or not val.get("pos"):
                continue
            examples.append({
                "text": val["text"],
                "pos": val["pos"]
            })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    cls_token = (tokenizer.config.cls_token_id,) * 3
    pad_token = (tokenizer.config.pad_token_id,) * 3
    
    features = []
    for example in tqdm(examples, desc="Converting features"):
        text = example["text"]
        pos_dict = example["pos"]
        
        words = text.split()
        input_ids = [cls_token]
        label_ids = [-100]
        
        for word in words:
            tag = pos_dict.get(word, "O")
            syllables = word.split('_')
            
            for i, syl in enumerate(syllables):
                bio_tag = f"B-{tag}" if i == 0 else f"I-{tag}"
                if tag == "O" or tag == "PU": 
                    bio_tag = "O"
                    
                l_id = LABEL_MAP.get(bio_tag, LABEL_MAP["O"])
                
                components = tokenizer.analyze(syl)
                if components:
                    initial, rhyme, tone = components
                    if rhyme in tokenizer.config.label2id:
                        input_ids.append((
                            tokenizer.config.label2id[initial] if initial else tokenizer.config.empty_token_id,
                            tokenizer.config.label2id[rhyme],
                            tokenizer.config.label2id[tone] if tone else tokenizer.config.empty_token_id 
                        ))
                        label_ids.append(l_id)
                    else:
                        input_ids.append((tokenizer.config.unk_token_id,) * 3)
                        label_ids.append(l_id)
                else:
                    for k, char in enumerate(syl):
                        input_ids.append((tokenizer.config.label2id.get(char, tokenizer.config.unk_token_id),) * 3)
                        label_ids.append(l_id if k == 0 else -100)
        
        input_ids = input_ids[:max_seq_length]
        label_ids = label_ids[:max_seq_length]
        
        attention_mask = [1] * len(input_ids)
        
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length
            label_ids += [-100] * padding_length

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids
        })
        
    return features

def create_dataloader(features, batch_size, is_training=True):
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f["label_ids"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


# ==========================================
# 3. ĐÁNH GIÁ (EVALUATION)
# ==========================================
def evaluate(model, dataloader, device):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_valid_tokens = 0, 0

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        preds = np.argmax(logits, axis=-1)

        eval_loss += loss.mean().item()
        
        # Tính Accuracy bỏ qua các token padding (-100)
        active_mask = label_ids != -100
        active_preds = preds[active_mask]
        active_labels = label_ids[active_mask]
        
        eval_accuracy += np.sum(active_preds == active_labels)
        nb_valid_tokens += len(active_labels)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_valid_tokens if nb_valid_tokens > 0 else 0
    
    return eval_loss, eval_accuracy


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- KHỞI TẠO MÔ HÌNH VỚI CONFIG MẶC ĐỊNH TỪ FILE PYTHON ---
    config = ViPhonBertConfig()
    tokenizer = ViPhonTokenizer(config) 
    model = ViPhonBertForTokenClassification(config, num_labels=NUM_LABELS).to(device)
    
    # --- LOAD TRỌNG SỐ TỪ FILE PTH TRỰC TIẾP ---
    logger.info(f"Loading Weights từ: {args.model_weight_path}")
    if os.path.exists(args.model_weight_path):
        checkpoint = torch.load(args.model_weight_path, map_location=device)
        
        # Xử lý: Nếu bạn lưu bằng checkpoint format (có key model_state_dict) thì lấy ra
        # Nếu lưu bằng model.state_dict() thì checkpoint chính là weights
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            logger.info("👉 Phát hiện định dạng Training Checkpoint Dictionary.")
        else:
            state_dict = checkpoint
            logger.info("👉 Phát hiện định dạng State Dict thuần túy.")
            
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Missing keys: {len(missing_keys)} (Bao gồm các layer phân loại mới khởi tạo)")
    else:
        logger.error(f"❌ Không tìm thấy file {args.model_weight_path}. Model sẽ train bằng Random Initialization.")

    # --- XỬ LÝ CACHE ---
    cache_train_path = os.path.join(args.data_dir, f"cached_niivtb_train_{args.max_seq_length}.pt")
    cache_dev_path = os.path.join(args.data_dir, f"cached_niivtb_dev_{args.max_seq_length}.pt")

    # Tập Train
    if os.path.exists(cache_train_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu Train, tải từ: {cache_train_path}")
        train_features = torch.load(cache_train_path)
    else:
        logger.info("❌ Không tìm thấy cache Train. Bắt đầu convert từ file JSON gốc...")
        train_examples = load_niivtb_data(os.path.join(args.data_dir, "train.json"))
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
        logger.info(f"💾 Lưu dữ liệu Train đã convert vào cache: {cache_train_path}")
        torch.save(train_features, cache_train_path)

    # Tập Dev
    if os.path.exists(cache_dev_path):
        logger.info(f"👉 Tìm thấy cache dữ liệu Dev, tải từ: {cache_dev_path}")
        dev_features = torch.load(cache_dev_path)
    else:
        logger.info("❌ Không tìm thấy cache Dev. Bắt đầu convert từ file JSON gốc...")
        dev_examples = load_niivtb_data(os.path.join(args.data_dir, "dev.json"))
        dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
        logger.info(f"💾 Lưu dữ liệu Dev đã convert vào cache: {cache_dev_path}")
        torch.save(dev_features, cache_dev_path)

    train_dataloader = create_dataloader(train_features, args.train_batch_size, is_training=True)
    dev_dataloader = create_dataloader(dev_features, args.eval_batch_size, is_training=False)

    t_total = len(train_dataloader) * args.epochs
    
    # --- TỐI ƯU OPTIMIZER ---
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * t_total), num_training_steps=t_total)
    scaler = GradScaler('cuda') 

    logger.info("***** Bắt đầu tiến trình Huấn luyện NIIVTB *****")
    best_acc = 0.0

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

        logger.info(f"***** Đánh giá chất lượng tập Dev (Epoch {epoch+1}) *****")
        eval_loss, eval_acc = evaluate(model, dev_dataloader, device)
        
        logger.info(f"Epoch {epoch+1} - Eval Loss: {eval_loss:.4f} - Eval Acc (Token-level): {eval_acc*100:.2f}%")

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_path = os.path.join(args.output_dir, "best_model_niivtb.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ LƯU KỶ LỤC TỐI ƯU MỚI: {best_acc*100:.2f}% tại {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa train.json, dev.json")
    parser.add_argument("--model_weight_path", type=str, required=True, help="Đường dẫn đến file .pth (Ví dụ: /network-volume/ViPhonBERT/viphon_bert_base_ep2.pth)")
    parser.add_argument("--output_dir", type=str, default="./niivtb_outputs", help="Thư mục lưu output")
    
    parser.add_argument("--max_seq_length", type=int, default=256, help="Độ dài tối đa của câu")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size đánh giá")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=10, help="Số epoch")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()