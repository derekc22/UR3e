# gymnasium_src/scripts/imitation_rl/custom_feature_extractor.py

import gymnasium as gym
import torch
import torch.nn as nn
import math
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    From the Pytorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # The stored positional encoding has shape [max_len, 1, d_model].
        # We need to add it to the input x, which has shape [batch_size, seq_len, d_model].
        # PyTorch's broadcasting will handle this. We first need to permute x.
        x = x.permute(1, 0, 2) # -> [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2) # -> [batch_size, seq_len, embedding_dim]
        return self.dropout(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using a Transformer Encoder for a sequence of observations.
    Designed to work with Gymnasium's FrameStack wrapper.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        d_model: int = 256,
        n_head: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim)
        
        # The observation space shape from FrameStack is (k, num_features)
        # We need the number of features for a single frame.
        self.history_len, single_frame_features = observation_space.shape
        
        # Input embedding layer
        self.embedding = nn.Linear(single_frame_features, d_model)

        # ** 1. ADD POSITIONAL ENCODING **
        self.pos_encoder = PositionalEncoding(d_model)

        # Define the Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layer to project to the final features_dim
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, history_len, single_frame_features)
        
        # Embed each frame in the sequence
        # -> (batch_size, history_len, d_model)
        embedded = self.embedding(observations)
        
        # Add positional information
        # -> (batch_size, history_len, d_model)
        with_pos = self.pos_encoder(embedded)

        # Pass the full sequence through the transformer
        # -> (batch_size, history_len, d_model)
        transformer_output = self.transformer_encoder(with_pos)
        
        # ** 3. EXTRACT FEATURES FROM THE SEQUENCE **
        # We take the output corresponding to the most recent frame (-1) as the representative feature for the whole sequence.
        # -> (batch_size, d_model)
        last_frame_features = transformer_output[:, -1, :]
        
        # Project to the final feature dimension
        # -> (batch_size, features_dim)
        return self.linear_out(last_frame_features)