from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from vocabs.viphon_tokenizer import ViPhonTokenizer, VietnameseEncodedTokens

import os
from tqdm import tqdm

PAD_TOKEN_ID = -100

def collate_fn(samples: list[VietnameseEncodedTokens]):
    # Extract tensors
    input_ids_list = [s.input_ids for s in samples]   # (seq_len, 3)
    labels_list = [s.labels for s in samples]

    # Pad sequences (batch_first=True → (bs, max_len, ...))
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=PAD_TOKEN_ID
    )

    # Attention mask: 1 where not PAD
    attention_mask = (input_ids[..., 0] != 0).long()

    return VietnameseEncodedTokens(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
    )

class ViPhonDataset(Dataset):
    def __init__(self, tokenizer: ViPhonTokenizer, corpus_dir, max_length=256):
        self.max_length = max_length
        self.corpus_dir = corpus_dir
        self.tokenizer = tokenizer
        self.txt_files = os.listdir(corpus_dir)
        self.total_line = 0
        for txt_file in tqdm(self.txt_files, desc="Loading data"):
            texts = open(os.path.join(corpus_dir, txt_file)).readlines()
            self.total_line += len(texts)

        self.LINE_PER_FILE = 10_000

    def __len__(self):
        return self.total_line

    def __getitem__(self, idx):
        # the default format for the corpus file of each line if subset_<idx>.txt
        subset_idx, line_idx = divmod(idx+1, self.LINE_PER_FILE)
        with open(os.path.join(self.corpus_dir, f"subset_{subset_idx}.txt")) as file:
            for line_ith, text in enumerate(file):
                if line_ith == line_idx - 1:
                    encoded_text = self.tokenizer(text)
                    break

        return encoded_text
