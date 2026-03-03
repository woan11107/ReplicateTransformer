from dataclasses import dataclass, field
import torch
@dataclass
class Config:
    # model config
    # encoder&decoder config
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1

    # attention config
    batch_size: int = 4 # How many batches per training step
    block_size: int = 16 # Length of the token chunk each batch
    d_model: int = 64 # The vector size of the token embeddings
    num_layers: int = 8 # Number of transformer blocks
    num_heads: int = 4 # Number of heads in Multi-head attention
    learning_rate: float = 1e-3
    dropout: float = 0.1
    max_iters: int = 5000 # Total of training iterations
    eval_interval: int = 50 # How often to evaluate the model
    eval_iters: int = 20 # How many iterations to average the loss over when evaluating the model
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_seed: int = 42

    # feedforward config
    hidden_layer_multiplier: int = 4 # The multiplier for the hidden layer size in the feedforward network (d_ff = d_model * hidden_layer_multiplier)

    # data config
    # 数据路径
    train_file: str = "datasets/sales_textbook.txt"
    val_file: str = None  # 可选，如果没有则从训练数据分割
    
    # 分词器
    tokenizer_type: str = "cl100k_base" # "gpt2"  # "cl100k_base" 是 tiktoken 的默认编码器，适用于大多数文本数据，尤其是英文文本。它的词汇表大小约为 100,000 个 token，能够处理各种常见的文本输入，包括标点符号、特殊字符和常用单词。对于一般的文本数据，使用 "cl100k_base" 可以提供更好的分词效果和更丰富的 token 表达能力。
    
    # 自动填充 (初始化后计算)
    vocab_size: int = field(init=False, default=0)
    train_size: int = field(init=False, default=0)
    val_size: int = field(init=False, default=0)
    
    # 设备
    device: str = field(init=False, default="cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        # set default device when initializing (PyTorch 2.0+)
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(self.device)