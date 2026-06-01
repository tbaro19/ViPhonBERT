import math
import torch
from torch.utils.data import DataLoader

from configs.viphon_bert_config import ViPhonBertConfig
from vocabs.viphon_tokenizer import ViPhonTokenizer
from data_utils.viphon_dataset import ViPhonDataset
# Đổi lại collate_fn của ViPhon nếu có, tạm thời giữ nguyên theo code của bạn
from data_utils.pinyin_dataset import collate_fn 

BS = 64
total_steps = 1_000_000

print("--- ĐANG KHỞI TẠO CONFIG & TOKENIZER ---")
config = ViPhonBertConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    max_length=1024, 
    type_vocab_size=1,
    is_decoder=False,
    add_cross_attention=False
)
tokenizer = ViPhonTokenizer(config)

print("\n--- ĐANG LOAD DATASET (Quá trình này có thể mất vài phút) ---")
dataset = ViPhonDataset(
    tokenizer=tokenizer, 
    corpus_dir="/network-volume/ViPhonBERT/data/Vietnamese-curated-corpus", 
    max_length=config.max_length
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=24,
    collate_fn=collate_fn
)

print("\n================ STATS REPORT ================")
print(f"1. Tổng số mẫu dữ liệu (Total Samples): {len(dataset)}")
print(f"2. Kích thước Batch Size (BS): {BS}")
print(f"3. Số lượng Batch trong DataLoader (len(dataloader)): {len(dataloader)}")
print(f"4. Mục tiêu Total Steps của bạn: {total_steps:,}")

# Tính thử theo toán tử // cũ của bạn
epochs_floor = total_steps // len(dataloader) if len(dataloader) > 0 else 0
print(f"5. Số Epoch tính theo code cũ (phép chia //): {epochs_floor}")

# Tính theo toán tử làm tròn lên chuẩn
if len(dataloader) > 0:
    epochs_ceil = math.ceil(total_steps / len(dataloader))
    print(f"6. Số Epoch gợi ý (làm tròn lên): {epochs_ceil}")
else:
    print("6. Cảnh báo: DataLoader trống (len = 0)!")
print("==============================================")