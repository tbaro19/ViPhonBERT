import argparse
import json
import logging
import os
import random
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.exceptions import UndefinedMetricWarning

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


# ==========================================
# 1. KIẾN TRÚC MÔ HÌNH (QUESTION ANSWERING)
# ==========================================
class ViPhonBertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = 2 
        self.hidden_size = config.hidden_size
        
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.fc_emb = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.bert.embeddings.word_embeddings = None

        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.qa_outputs.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.qa_outputs.bias is not None:
            self.qa_outputs.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
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
        
        logits = self.qa_outputs(sequence_output) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous() 
        end_logits = end_logits.squeeze(-1).contiguous()     

        total_loss = None
        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss, start_logits, end_logits


# ==========================================
# 2. XỬ LÝ DỮ LIỆU VINEWSQA
# ==========================================
def load_vinewsqa_data(data_path):
    examples = []
    if not os.path.exists(data_path):
        logger.warning(f"Tệp dữ liệu không tồn tại: {data_path}")
        return examples

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        # ViNewsQA có cấu trúc root là "data"
        data = dataset.get("data", [])
        
        for article in data:
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):
                    answer_text = ""
                    answers = qa.get("answers", [])
                    
                    # Lấy câu trả lời đầu tiên làm ground truth
                    if len(answers) > 0:
                        answer_text = answers[0].get("text", "")
                    
                    examples.append({
                        "id": qa.get("id", ""),
                        "question": qa.get("question", ""),
                        "context": context,
                        "answer_text": answer_text
                    })
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length):
    pad_token = (tokenizer.config.pad_token_id,) * 3
    features = []
    
    for example in tqdm(examples, desc="Converting features"):
        q_ids = tokenizer.encode(example["question"]).tolist() 
        c_ids = tokenizer.encode(example["context"]).tolist()
        
        # Bỏ [CLS] của context khi nối vào sau question
        if c_ids[0] == (tokenizer.config.cls_token_id,) * 3:
            c_ids = c_ids[1:]
            
        input_ids = q_ids + c_ids
        input_ids = input_ids[:max_seq_length]
        attention_mask = [1] * len(input_ids)
        
        start_position = 0
        end_position = 0
        
        if example["answer_text"]:
            ans_ids = tokenizer.encode(example["answer_text"]).tolist()
            if ans_ids[0] == (tokenizer.config.cls_token_id,) * 3:
                ans_ids = ans_ids[1:]
            
            ans_len = len(ans_ids)
            # Tìm vị trí token của câu trả lời trong chuỗi input_ids
            for i in range(len(q_ids), len(input_ids) - ans_len + 1):
                if input_ids[i:i+ans_len] == ans_ids:
                    start_position = i
                    end_position = i + ans_len - 1
                    break
        
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids += [pad_token] * padding_length
            attention_mask += [0] * padding_length

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_position": start_position,
            "end_position": end_position
        })
        
    return features

def create_dataloader(features, batch_size, is_training=True):
    if not features:
        return None
    
    all_input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f["start_position"] for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f["end_position"] for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_start_positions, all_end_positions)
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)


# ==========================================
# 3. ĐÁNH GIÁ (SPAN F1 VÀ EXACT MATCH)
# ==========================================
def evaluate(model, dataloader, device):
    if dataloader is None:
        return 0, 0, 0
        
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    
    correct_exact_match = 0
    total_f1 = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, start_positions, end_positions = batch

        with torch.no_grad():
            loss, start_logits, end_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )

        eval_loss += loss.item()
        
        start_preds = torch.argmax(start_logits, dim=-1).cpu().numpy()
        end_preds = torch.argmax(end_logits, dim=-1).cpu().numpy()
        start_trues = start_positions.cpu().numpy()
        end_trues = end_positions.cpu().numpy()

        for i in range(len(start_preds)):
            sp, ep = start_preds[i], end_preds[i]
            st, et = start_trues[i], end_trues[i]

            # Exact Match
            if sp == st and ep == et:
                correct_exact_match += 1

            # Tính F1 Score cho từng Span (Mức độ giao thoa token)
            pred_span = set(range(sp, ep + 1)) if sp <= ep and sp > 0 else set()
            true_span = set(range(st, et + 1)) if st <= et and st > 0 else set()

            if len(pred_span) == 0 and len(true_span) == 0:
                f1 = 1.0
            elif len(pred_span) == 0 or len(true_span) == 0:
                f1 = 0.0
            else:
                num_same = len(pred_span.intersection(true_span))
                if num_same == 0:
                    f1 = 0.0
                else:
                    precision = 1.0 * num_same / len(pred_span)
                    recall = 1.0 * num_same / len(true_span)
                    f1 = (2 * precision * recall) / (precision + recall)
            
            total_f1 += f1
            total_samples += 1
            
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps if nb_eval_steps > 0 else 0
    exact_match_acc = correct_exact_match / total_samples if total_samples > 0 else 0
    
    # avg_f1 chính là Average F1 (Macro-average F1 over examples) chuẩn của bài toán QA
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0
    
    return eval_loss, exact_match_acc, avg_f1


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
    
    model = ViPhonBertForQuestionAnswering(config).to(device)
    
    logger.info(f"Loading Weights từ: {weights_path}")
    if os.path.exists(weights_path):
        if weights_path.endswith('.safetensors'):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.info("❌ Không tìm thấy checkpoint. Train từ Random Initialization.")

    # ---------------------------------------------
    # 4.1 XỬ LÝ CACHE CHO VINEWSQA
    # ---------------------------------------------
    cache_train_path = os.path.join(args.data_dir, f"cached_vinewsqa_train_{args.max_seq_length}.pt")
    cache_dev_path = os.path.join(args.data_dir, f"cached_vinewsqa_dev_{args.max_seq_length}.pt")
    cache_test_path = os.path.join(args.data_dir, f"cached_vinewsqa_test_{args.max_seq_length}.pt")

    if os.path.exists(cache_train_path):
        train_features = torch.load(cache_train_path)
    else:
        train_examples = load_vinewsqa_data(os.path.join(args.data_dir, "train.json"))
        train_features = convert_examples_to_features(train_examples, tokenizer, args.max_seq_length)
        torch.save(train_features, cache_train_path)

    if os.path.exists(cache_dev_path):
        dev_features = torch.load(cache_dev_path)
    else:
        dev_examples = load_vinewsqa_data(os.path.join(args.data_dir, "dev.json"))
        dev_features = convert_examples_to_features(dev_examples, tokenizer, args.max_seq_length)
        torch.save(dev_features, cache_dev_path)

    if os.path.exists(cache_test_path):
        test_features = torch.load(cache_test_path)
    else:
        test_examples = load_vinewsqa_data(os.path.join(args.data_dir, "test.json"))
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

    logger.info("***** Bắt đầu tiến trình Huấn luyện ViNewsQA *****")
    best_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model_vinewsqa.pt")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, start_positions, end_positions = batch

                optimizer.zero_grad()

                with autocast('cuda'):
                    loss, _, _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        start_positions=start_positions,
                        end_positions=end_positions
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
        eval_loss, eval_em, eval_f1 = evaluate(model, dev_dataloader, device)
        
        logger.info(f"Epoch {epoch+1} - Eval Loss: {eval_loss:.4f} | Exact Match: {eval_em*100:.2f}% | Avg Span F1: {eval_f1*100:.2f}%")

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✨ LƯU KỶ LỤC TỐI ƯU MỚI: Avg Span F1 = {best_f1*100:.2f}% tại {best_model_path}")

    # ---------------------------------------------
    # 4.2 ĐÁNH GIÁ TRÊN TẬP TEST SAU KHI TRAIN XONG
    # ---------------------------------------------
    logger.info("==================================================")
    logger.info("***** Bắt đầu chấm điểm trên tập Test.json *****")
    if test_dataloader is not None and os.path.exists(best_model_path):
        logger.info("Đang nạp lại trọng số tốt nhất...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        test_loss, test_em, test_f1 = evaluate(model, test_dataloader, device)
        logger.info(f"🎯 KẾT QUẢ TEST CUỐI CÙNG - Loss: {test_loss:.4f}")
        logger.info(f"🎯 Exact Match: {test_em*100:.2f}%")
        logger.info(f"🎯 Avg Span F1: {test_f1*100:.2f}%")
    else:
        logger.info("⚠️ Bỏ qua bước Test (không tìm thấy test.json hoặc test_dataloader trống).")
    logger.info("==================================================")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa train.json, dev.json, test.json")
    parser.add_argument("--init_checkpoint", type=str, required=True, help="Thư mục weights gốc")
    parser.add_argument("--output_dir", type=str, default="./outputs_vinewsqa", help="Thư mục lưu output")
    
    parser.add_argument("--max_seq_length", type=int, default=384, help="Độ dài tối đa của câu")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size huấn luyện")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size đánh giá/test")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()