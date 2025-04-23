import torch
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=768, dropout=0.1):
        """
        Vision encoder to process ResNet-50 features before feeding to language model.
        
        Args:
            input_dim (int): Input dimension from ResNet-50 (2048)
            hidden_dim (int): Hidden dimension of the encoder
            output_dim (int): Output dimension (should match Llama's embedding dimension)
            dropout (float): Dropout probability
        """
        super(VisionEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
    def forward(self, x):
        """
        Forward pass of the vision encoder.
        
        Args:
            x (torch.Tensor): Tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Encoded representation of shape [batch_size, output_dim]
        """
        return self.encoder(x)

def create_vision_encoder(input_dim=2048, hidden_dim=512, output_dim=768, dropout=0.1):
    return VisionEncoder(input_dim, hidden_dim, output_dim, dropout) 