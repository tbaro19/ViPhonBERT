import argparse
import json
import logging
import os
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

import torch

# Import trực tiếp các cấu hình và class từ project của bạn
from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer

# Import các hàm và hằng số cần thiết từ file train (đảm bảo file tên là finetune_viocd.py)
from finetune_viocd import (
    ViPhonBertForSequenceClassification, 
    load_viocd_data, 
    convert_examples_to_features, 
    create_dataloader,
    NUM_LABELS
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_test_set(model, dataloader, device):
    """
    Hàm đánh giá chuyên biệt để tính Macro F1 cho bài toán Sequence Classification.
    """
    model.eval()
    all_preds = []
    all_labels = []

    logger.info("Tiến hành dự đoán trên tập Test...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            _, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Lấy class có xác suất cao nhất (0 hoặc 1)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Tính Accuracy và Macro F1
    accuracy = np.sum(all_preds == all_labels) / len(all_labels) if len(all_labels) > 0 else 0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return accuracy, macro_f1

def main():
    parser = argparse.ArgumentParser()
    # Các tham số theo đúng setup bạn yêu cầu
    parser.add_argument("--data_dir", type=str, required=True, help="Thư mục chứa test.json")
    parser.add_argument("--config_dir", type=str, required=True, help="Thư mục chứa config.json của model gốc")
    parser.add_argument("--model_path", type=str, required=True, help="Đường dẫn tới file .pt đã lưu kỷ lục")
    
    # Các tham số ẩn (có giá trị mặc định giống file train)
    parser.add_argument("--max_seq_length", type=int, default=128, help="Độ dài tối đa của câu")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size đánh giá")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Config & Tokenizer
    logger.info(f"Loading Config từ: {args.config_dir}")
    config_path = os.path.join(args.config_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = ViPhonBertConfig(**config_dict)
    tokenizer = ViPhonTokenizer(config) 
    
    # 2. Khởi tạo Model & Load Trọng số (.pt)
    model = ViPhonBertForSequenceClassification(
        config=config, 
        num_labels=NUM_LABELS
    ).to(device)
    
    logger.info(f"Loading Weights (Kỷ lục F1) từ: {args.model_path}")
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logger.info("Tải trọng số thành công!")
    else:
        logger.error(f"❌ Không tìm thấy file trọng số tại: {args.model_path}")
        return

    # 3. Load Dữ liệu Test
    cache_test_path = os.path.join(args.data_dir, f"cached_viocd_test_{args.max_seq_length}.pt")
    if os.path.exists(cache_test_path):
        logger.info(f"Loading cached test features từ {cache_test_path}")
        test_features = torch.load(cache_test_path)
    else:
        logger.info("Chưa có cache, đang tiến hành xử lý test.json...")
        test_examples = load_viocd_data(os.path.join(args.data_dir, "test.json"))
        if not test_examples:
            logger.error("❌ Không tìm thấy hoặc test.json rỗng!")
            return
        test_features = convert_examples_to_features(test_examples, tokenizer, args.max_seq_length)
        torch.save(test_features, cache_test_path)

    test_dataloader = create_dataloader(test_features, args.eval_batch_size, is_training=False)

    # 4. Đánh giá
    logger.info("==================================================")
    test_acc, macro_f1 = evaluate_test_set(model, test_dataloader, device)
    
    logger.info(f"🎯 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    logger.info(f"   -> Accuracy: {test_acc * 100:.2f}%")
    logger.info(f"   -> Macro F1: {macro_f1 * 100:.2f}%")
    logger.info("==================================================")

if __name__ == "__main__":
    main()