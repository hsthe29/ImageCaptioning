import torch
from torch import nn

from .decoder import Decoder


class CaptionGenerator(nn.Module):
    def __init__(self, *,
                 max_position_embeddings,
                 vocab_size,
                 num_layers,
                 hidden_size,
                 intermediate_size,
                 image_feature_size):
        super(CaptionGenerator, self).__init__()
        
        self.decoder = Decoder(max_position_embeddings=max_position_embeddings,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               num_attention_heads=4,
                               hidden_size=hidden_size,
                               intermediate_size=intermediate_size)
        
        self.resize = nn.Linear(image_feature_size, hidden_size)
        
        self.output_fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image_feature, input_ids, attention_mask):
        image = self.resize(image_feature)
        
        decoder_output = self.decoder(image, input_ids, attention_mask)
        logits = self.output_fc(decoder_output)
        return logits
