import torch
import torch.nn as nn
from ..utils.encoding import PositionalEncoder

class NeRF(nn.Module):
    """
    Neural Radiance Field model.
    """
    def __init__(
        self,
        d_input: int = 3,
        n_layers: int = 8,
        d_filter: int = 256,
        skip: tuple = (4,),
        d_view_dirs: int = 3,
        n_freqs_xyz: int = 10,
        n_freqs_dirs: int = 4
    ):
        """
        Args:
            d_input (int): Input dimension for spatial coordinates (x, y, z).
            n_layers (int): Number of layers in the main MLP.
            d_filter (int): Number of hidden units in each layer.
            skip (tuple): Layer indices for skip connections.
            d_view_dirs (int): Input dimension for viewing directions.
            n_freqs_xyz (int): Number of frequencies for positional encoding of coordinates.
            n_freqs_dirs (int): Number of frequencies for positional encoding of directions.
        """
        super().__init__()
        self.n_layers = n_layers
        self.skip = skip
        
        # Positional encoders
        self.encoder_xyz = PositionalEncoder(d_input, n_freqs_xyz)
        self.encoder_dirs = PositionalEncoder(d_view_dirs, n_freqs_dirs)
        
        d_xyz_encoded = self.encoder_xyz.d_output
        d_dirs_encoded = self.encoder_dirs.d_output

        # MLP for spatial features
        self.mlp_xyz = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(d_xyz_encoded, d_filter)
            elif i in skip:
                layer = nn.Linear(d_filter + d_xyz_encoded, d_filter)
            else:
                layer = nn.Linear(d_filter, d_filter)
            self.mlp_xyz.append(layer)
        
        # Feature and density prediction layers
        self.fc_feature = nn.Linear(d_filter, d_filter)
        self.fc_density = nn.Linear(d_filter, 1)
        
        # Color prediction layer
        self.fc_rgb = nn.Sequential(
            nn.Linear(d_filter + d_dirs_encoded, d_filter // 2),
            nn.ReLU(),
            nn.Linear(d_filter // 2, 3)
        )

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NeRF model.

        Args:
            x (torch.Tensor): Sampled points' coordinates (batch_size, n_samples, 3).
            view_dirs (torch.Tensor): Sampled points' view directions (batch_size, n_samples, 3).

        Returns:
            torch.Tensor: Predicted RGB and density (batch_size, n_samples, 4).
        """
        # Encode inputs
        x_encoded = self.encoder_xyz(x)
        dirs_encoded = self.encoder_dirs(view_dirs)
        
        # Process spatial coordinates
        h = x_encoded
        for i, layer in enumerate(self.mlp_xyz):
            h = torch.relu(layer(h))
            if i in self.skip:
                h = torch.cat([h, x_encoded], dim=-1)
                
        # Get density
        density = self.fc_density(h)
        
        # Get RGB color
        feature = self.fc_feature(h)
        rgb_input = torch.cat([feature, dirs_encoded], dim=-1)
        rgb = torch.sigmoid(self.fc_rgb(rgb_input))
        
        return torch.cat([rgb, density], dim=-1) 