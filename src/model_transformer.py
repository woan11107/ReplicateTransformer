import math
import torch
import torch.nn as nn
from config import Config

class AttentionLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_size = self.d_model // self.num_heads
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # initialize linear layers for query, key, value, and output projections
        # these should be defined in __init__ to ensure they are registered as parameters of the model and will be updated during training
        self.Wq = nn.Linear(self.d_model, self.d_model)
        self.Wk = nn.Linear(self.d_model, self.d_model)
        self.Wv = nn.Linear(self.d_model, self.d_model)
        self.Wo = nn.Linear(self.d_model, self.d_model)

        self.ln = nn.LayerNorm(self.d_model)

        # register causal mask
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        # register as buffer so it will be moved to the correct device with the model, but not saved in the state dict since it's deterministic and can be recreated
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x ~ (batch_size, block_size, d_model)
        batch_size, block_size, d_model = x.size()
        
        # calculate q, k, v for all heads ~ (batch_size, num_heads, block_size, head_size)
        q = self.Wq(x).view(batch_size, block_size, self.num_heads, self.head_size).transpose(1, 2)
        k = self.Wk(x).view(batch_size, block_size, self.num_heads, self.head_size).transpose(1, 2)
        v = self.Wv(x).view(batch_size, block_size, self.num_heads, self.head_size).transpose(1, 2)

        # calculate attention score : Q@K^T / sqrt(d_k) ~ (batch_size, num_heads, block_size, block_size)
        attention_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # apply causal mask to prevent attending to future tokens
        # ~ ： not operator, turn true in the mask to false and false to true
        attention_score = attention_score.masked_fill(~self.causal_mask[:block_size, :block_size], float('-inf'))

        # apply softmax to get attention weights
        attention_weights = torch.softmax(attention_score, dim = -1)

        # calculate attention output : attention_weights @ V
        attention_output = torch.matmul(attention_weights, v) # (batch_size, num_heads, block_size, head_size)

        # concatenate different heads and project back to d_model dimensions
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, block_size, d_model)

        # apply output linear transformation
        attention_output = self.Wo(attention_output)

        # add residual connection and layer norm
        attention_output = self.ln(attention_output + x)
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.hidden_layer_multiplier),
            nn.ReLU(),
            nn.Linear(config.d_model * config.hidden_layer_multiplier, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.attention = AttentionLayer(config)
        self.feed_forward = FeedForward(config)
        self.n_encoder_layers = config.n_encoder_layers

    def forward(self, x):
        for _ in range(self.n_encoder_layers):
            x = self.attention(x)
            x = self.feed_forward(x)
        return x