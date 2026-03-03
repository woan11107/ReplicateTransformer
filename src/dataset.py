import torch
from torch.utils.data import Dataset
import tiktoken
from config import Config

class Tokenizer:
    def __init__(self, tokenizer_type: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(tokenizer_type)
        self.vocab_size = self.tokenizer.n_vocab
    
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    @property
    def max_token_value(self) -> int:
        """最大 token ID 值"""
        return self.vocab_size - 1

class TextDataset(Dataset):
    def __init__(
        self, 
        tokenized_data: list[int], 
        block_size: int,
        name: str = "train"
    ):
        self.data = tokenized_data
        self.block_size = block_size
        self.name = name
        self.size = len(self.data) - block_size  # 可用样本数
    
    def __len__(self) -> int:
        return max(0, self.size)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入：[idx, idx+block_size]
        # 目标：[idx+1, idx+block_size+1]
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    @property
    def max_token_value(self) -> int:
        return max(self.data) if self.data else 0