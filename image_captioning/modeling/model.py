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
        
        self.hidden_units = hidden_units
        
        self.resize_features = keras.layers.Dense(sentence_length * embedding_sizes[0], activation='relu')
        self.reshape = keras.layers.Reshape(target_shape=(sentence_length, embedding_sizes[0]))
        self.scale = keras.layers.Dense(hidden_units, activation='relu')
        
        self.embedding = keras.layers.Embedding(vocab_size, embedding_sizes[1], mask_zero=True)
        self.lstm = keras.layers.LSTM(hidden_units,
                                      return_sequences=True,
                                      return_state=True,
                                      recurrent_initializer='glorot_uniform')
        self.attention = Attention(128)
        
        self.dense = keras.layers.Dense(vocab_size)
    
    def init_state(self, batch_size):
        return [
            tf.zeros([batch_size, self.hidden_units]),
            tf.zeros([batch_size, self.hidden_units])
        ]
    
    def call(self, inputs):
        image_feats, tokens = inputs
        q_mask = input != 0
        batch_size = tf.shape(image_feats)[0]
        img_feats = self.resize_features(image_feats)
        img_feats = self.reshape(img_feats)
        img_feats = self.scale(img_feats)
        embedding = self.embedding(tokens)
        first_state = self.init_state(batch_size)
        query, h_s, c_s = self.lstm(embedding, initial_state=first_state)
        context, _ = self.attention(query, img_feats, q_mask)
        logits = self.dense(context)
        
        return logits
    
    def get_image_context(self, image_features):
        img_feats = self.resize_features(image_features)
        img_feats = self.reshape(img_feats)
        img_feats = self.scale(img_feats)
        return img_feats
    
    def next_tokens(self, in_tokens, image_context, q_mask, state):
        embedding = self.embedding(in_tokens)
        query, h_s, c_s = self.lstm(embedding, initial_state=state)
        context, attn_scores = self.attention(query, image_context, q_mask)
        logits = self.dense(context)
        
        return logits, attn_scores, [h_s, c_s]
