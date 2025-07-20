import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.encoding import PositionalEncoder

class NeRFModule(nn.Module):
    """
    A simple NeRF MLP model.
    """
    def __init__(self, config, condition_dim=0):
        super().__init__()
        self.D = config['model']['nerf']['num_layers']
        self.W = config['model']['nerf']['embedding_dim']
        self.use_viewdirs = config['model']['nerf']['use_viewdirs']
        self.condition_dim = condition_dim
        
        # Positional encoding for 3D coordinates
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=10)
        self.input_ch = self.pos_encoder.d_output
        
        # Positional encoding for view directions
        if self.use_viewdirs:
            self.dir_encoder = PositionalEncoder(d_input=3, n_freqs=4)
            self.input_ch_views = self.dir_encoder.d_output
        
        # MLP layers
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + 
            [nn.Linear(self.W, self.W) if i != 4 else nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D - 1)]
        )
        
        # Modified part to include conditioning
        if self.condition_dim > 0:
            self.feature_linear = nn.Linear(self.W + self.condition_dim, self.W)
        else:
            self.feature_linear = nn.Linear(self.W, self.W)
            
        self.alpha_linear = nn.Linear(self.W, 1)
        
        if self.use_viewdirs:
            self.rgb_linear = nn.Linear(self.W // 2 + self.input_ch_views, self.W // 2)
        else:
            self.rgb_linear = nn.Linear(self.W, self.W // 2)
            
        self.output_linear = nn.Linear(self.W // 2, 3)

    def forward(self, x, view_dirs=None, condition=None):
        # x: (B, N_rays, N_samples, 3) - 3D points
        # view_dirs: (B, N_rays, 3) - view directions
        # condition: (B, N_rays, N_samples, D_feature) - Conditional features
        
        # Positional encoding
        encoded_pts = self.pos_encoder(x)
        h = encoded_pts
        
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i == 4: # Skip connection
                h = torch.cat([encoded_pts, h], -1)

        # Apply condition
        if self.condition_dim > 0:
            assert condition is not None
            h = torch.cat([h, condition], dim=-1)
        
        feature = self.feature_linear(h)
        alpha = self.alpha_linear(feature) # Use feature after conditioning
        
        if self.use_viewdirs:
            assert view_dirs is not None
            # Prepare view directions
            view_dirs_expanded = view_dirs.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
            encoded_dirs = self.dir_encoder(view_dirs_expanded)
            
            # Concatenate features and encoded directions
            h = torch.cat([feature, encoded_dirs], -1)
            h = self.rgb_linear(h)
            h = F.relu(h)
        else:
            h = self.rgb_linear(feature)

        rgb = self.output_linear(h) # (B, N_rays, N_samples, 3)
        
        outputs = torch.cat([rgb, alpha], -1) # (B, N_rays, N_samples, 4)
        return outputs

if __name__ == '__main__':
    from configs.default_config import config
    
    # Add condition_dim for test
    condition_dim = 128
    config['model']['nerf']['condition_dim'] = condition_dim
    
    model = NeRFModule(config, condition_dim=condition_dim).cuda()
    
    points = torch.randn(2, 1024, 64, 3).cuda()
    view_dirs = torch.randn(2, 1024, 3).cuda()
    condition = torch.randn(2, 1024, 64, condition_dim).cuda()
    
    output = model(points, view_dirs, condition=condition)
    print(f"Conditioned NeRF output shape: {output.shape}") 