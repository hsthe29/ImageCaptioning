import torch
from torch import nn

from .layers import Embeddings, DecoderLayer


class Decoder(nn.Module):
    def __init__(self, *,
                 max_position_embeddings,
                 vocab_size,
                 num_layers,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size,):
        super().__init__()
        
        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            hidden_size=hidden_size,
            pad_id=0)
        
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size))
    