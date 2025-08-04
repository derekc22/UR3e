import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    The positional encodings have the same dimension as the embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        # Calculate sine for even indices and cosine for odd indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer, so it's part of the model's state but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        # Add positional encoding to the input tensor
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    A standard Transformer Encoder composed of N identical layers.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input Embedding Layer
        # In a real application, you'd have an nn.Embedding layer here.
        # For this example, we assume embeddings are pre-computed.
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Create a single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # PyTorch default is (seq_len, batch_size, d_model)
        )
        
        # Stack N encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: the sequence to the encoder (required).
                 Shape: [seq_len, batch_size, d_model]
            src_mask: the mask for the src sequence (optional).
                      Shape: [seq_len, seq_len]
        
        Returns:
            Output tensor of shape [seq_len, batch_size, d_model]
        """
        # 1. Add positional encoding
        src = self.pos_encoder(src)
        
        # 2. Pass through the encoder stack
        output = self.transformer_encoder(src, mask=src_mask)
        
        return output


if __name__ == '__main__':
    # Model hyperparameters
    d_model = 512       # Dimension of the model (embeddings)
    nhead = 8           # Number of attention heads
    num_layers = 6      # Number of encoder layers
    dim_feedforward = 2048 # Dimension of the feed-forward network
    dropout = 0.1       # Dropout rate

    # Create the Transformer Encoder model
    encoder = TransformerEncoder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    # Create some dummy input data
    seq_len = 35
    batch_size = 10
    # Input tensor shape: (sequence_length, batch_size, model_dimension)
    src_input = torch.rand(seq_len, batch_size, d_model)

    # Get the model output
    output = encoder(src_input)

    print("Input shape:", src_input.shape)
    print("Output shape:", output.shape)
    # Expected output shape: (35, 10, 512)
    