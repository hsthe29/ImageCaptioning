import math

import torch
from torch import nn
from torch.nn import functional as tf


class Embeddings(nn.Module):
    def __init__(self, *,
                 vocab_size,
                 max_position_embeddings,
                 hidden_size,
                 pad_id):
        super(Embeddings, self).__init__()
        self.ids_embedding = nn.Embedding(vocab_size,
                                          hidden_size,
                                          padding_idx=pad_id)
        
        self.pos_embedding = nn.Embedding(max_position_embeddings,
                                          hidden_size)
        self.scaling = math.sqrt(hidden_size)
    
    def forward(self, input_ids):
        ids_embedding = self.ids_embedding(input_ids)
        pos_embedding = self.pos_embedding(torch.arange(input_ids.shape[-1]).unsqueeze(0).to(input_ids.device))
        pos_embedding = pos_embedding / self.scaling
        return ids_embedding + pos_embedding


class AddNormLayer(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps):
        super(AddNormLayer, self).__init__()
        
        self.layer_norm = nn.LayerNorm(hidden_size,
                                       eps=layer_norm_eps)
    
    def forward(self, x, x_residual):
        return self.layer_norm(x + x_residual)


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size,
                 intermediate_size):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, inputs):
        x = tf.gelu(self.fc1(inputs))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.scaling = math.sqrt(hidden_size // num_attention_heads)
    
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: torch.FloatTensor | None = None):
        scores = torch.matmul(query, key.transpose(-1, -2))
        
        scaled_scores = scores / self.scaling
        
        if mask is not None:
            scaled_scores = scaled_scores + mask
        
        weights = tf.softmax(scaled_scores, dim=-1)
        weights = self.dropout(weights)
        
        outputs = torch.matmul(weights, value)
        return outputs, weights


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(SelfAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        
        assert hidden_size % self.num_attention_heads == 0
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.base_attention = ScaledDotProductAttention(num_attention_heads, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def split_heads(self, x):
        # [N, T, D] -> [N, T, h, S] -> [N, h, T, S]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def merge_heads(self, x):
        # [N, h, T, S] -> [N, T, h, S] -> [N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (-1,)
        return x.view(new_x_shape)
    
    def forward(self, inputs, attention_mask):
        query = self.split_heads(self.q_proj(inputs))
        key = self.split_heads(self.k_proj(inputs))
        value = self.split_heads(self.v_proj(inputs))
        
        outputs, weights = self.base_attention(query, key, value, attention_mask)
        
        outputs = self.merge_heads(outputs)
        outputs = self.out_proj(outputs)
        
        return outputs


class CrossAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super(CrossAttention, self).__init__()
        
        self.num_attention_heads = num_attention_heads
        
        assert hidden_size % self.num_attention_heads == 0
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.base_attention = ScaledDotProductAttention(num_attention_heads, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def split_heads(self, x):
        # [N, T, D] -> [N, T, h, S] -> [N, h, T, S]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def merge_heads(self, x):
        # [N, h, T, S] -> [N, T, h, S] -> [N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (-1,)
        return x.view(new_x_shape)
    
    def forward(self, q, k, attention_mask):
        query = self.split_heads(self.q_proj(q))
        key = self.split_heads(self.k_proj(k))
        value = self.split_heads(self.v_proj(k))
        
        outputs, weights = self.base_attention(query, key, value, attention_mask)
        
        outputs = self.merge_heads(outputs)
        outputs = self.out_proj(outputs)
        
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, *,
                 num_attention_heads,
                 hidden_size,
                 intermediate_size):
        super(DecoderLayer, self).__init__()
        self.self = SelfAttention(num_attention_heads, hidden_size)
        self.cross = CrossAttention(num_attention_heads, hidden_size)
        
        self.ffl = FeedForwardLayer(hidden_size, intermediate_size)
        
        self.self_add_norm = AddNormLayer(hidden_size, 1e-6)
        self.cross_add_norm = AddNormLayer(hidden_size, 1e-6)
        self.ffl_add_norm = AddNormLayer(hidden_size, 1e-6)
    
    def forward(self,
                encoder_context,
                inputs,
                attention_mask):
        attn_outputs = self.self(inputs, attention_mask)
        x = self.self_add_norm(attn_outputs, inputs)
        
        attn_outputs, cross_weights = self.cross(
            q_input=x,
            k_input=encoder_context,
            attention_mask=None
        )
        x = self.cross_add_norm(attn_outputs, x)
        
        ffl_outputs = self.ffl(x)
        x = self.ffl_add_norm(ffl_outputs, x)
        return x
    