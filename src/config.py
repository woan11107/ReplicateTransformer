from dataclasses import dataclass
import torch
@dataclass
class Config:
    batch_size: int = 4 # How many batches per training step
    context_length: int = 16 # Length of the token chunk each batch
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