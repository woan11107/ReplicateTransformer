from torch.utils.data import DataLoader as TorchDataLoader
from dataset import TextDataset, Tokenizer
from config import Config

class DataPipeline:
    """
    完整的数据管道：加载 → 分词 → 创建 Dataset → 提供 DataLoader
    """
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = Tokenizer(config.tokenizer_type)
        
        # 关键状态变量
        self.vocab_size = self.tokenizer.vocab_size
        self.train_dataset: TextDataset | None
        self.val_dataset: TextDataset | None
        self.train_loader: TorchDataLoader | None
        self.val_loader: TorchDataLoader | None
        
        # 数据统计
        self.train_size = 0
        self.val_size = 0
    
    def load_and_prepare(self, train_file: str, val_file: str | None):
        """加载文件并准备数据集"""
        # 1. 加载训练数据
        train_text = self._load_file(train_file)
        train_tokens = self.tokenizer.encode(train_text)
        self.train_dataset = TextDataset(train_tokens, self.config.block_size, "train")
        self.train_size = len(train_tokens)
        
        # 2. 加载验证数据 (可选)
        if val_file:
            val_text = self._load_file(val_file)
            val_tokens = self.tokenizer.encode(val_text)
            self.val_dataset = TextDataset(val_tokens, self.config.block_size, "val")
            self.val_size = len(val_tokens)
        else:
            # 如果没有验证集，从训练集分割 10%
            split = int(len(train_tokens) * 0.9)
            self.train_dataset = TextDataset(train_tokens[:split], self.config.block_size, "train")
            self.val_dataset = TextDataset(train_tokens[split:], self.config.block_size, "val")
            self.train_size = split
            self.val_size = len(train_tokens) - split
        
        # 3. 创建 DataLoader
        self.train_loader = TorchDataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,  # Windows 建议设为 0
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        self.val_loader = TorchDataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        # 4. 更新配置
        self.config.vocab_size = self.vocab_size
        self.config.train_size = self.train_size
        self.config.val_size = self.val_size
    
    def _load_file(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_train_iterator(self):
        """获取训练数据迭代器"""
        if self.train_loader is None:
            raise ValueError("请先调用 load_and_prepare()")
        return iter(self.train_loader)
    
    def get_val_iterator(self):
        """获取验证数据迭代器"""
        if self.val_loader is None:
            raise ValueError("请先调用 load_and_prepare()")
        return iter(self.val_loader)
    
    def get_stats(self) -> dict:
        """获取数据统计信息"""
        return {
            'vocab_size': self.vocab_size,
            'train_size': self.train_size,
            'val_size': self.val_size,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'max_token_value': self.tokenizer.max_token_value,
            'block_size': self.config.block_size,
            'batch_size': self.config.batch_size
        }