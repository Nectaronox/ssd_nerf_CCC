import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.encoding import PositionalEncoder

class DiffusionModule(nn.Module):
        
    def __init__(self, config):
        super().__init__()
        self.time_steps = config['model']['diffusion']['time_steps']
        self.feature_dim = config['model']['diffusion']['feature_dim']
    
        self.time_embedding = nn.Embedding(self.time_steps, self.feature_dim)
        # Use the unified PositionalEncoder from utils
        self.scene_time_encoder = PositionalEncoder(d_input=1, n_freqs=10)
        scene_time_dim = self.scene_time_encoder.d_output
        
        # Input: (B, N, 3 + diffusion_feature_dim + scene_time_dim)
        self.fc1 = nn.Linear(3 + self.feature_dim + scene_time_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, self.feature_dim)

    def forward(self, noisy_points, diffusion_t, scene_t=None):
        """
        Args:
            noisy_points (torch.Tensor): (B, N, 3) tensor of noisy point coordinates.
            diffusion_t (torch.Tensor): (B,) tensor of diffusion time steps.
            scene_t (torch.Tensor): (B, 1) tensor of scene time steps.
        
        Returns:
            torch.Tensor: Predicted noise or feature map (B, N, feature_dim).
        """
    
        diff_time_emb = self.time_embedding(diffusion_t).unsqueeze(1).expand(-1, noisy_points.shape[1], -1)
        
        if scene_t is not None:
            # PositionalEncoder expects input with proper dimensions
            scene_time_emb = self.scene_time_encoder(scene_t).unsqueeze(1).expand(-1, noisy_points.shape[1], -1)
            x = torch.cat([noisy_points, diff_time_emb, scene_time_emb], dim=-1)
        else:
            # Create zero padding with correct dimensions
            zero_scene_emb = torch.zeros(noisy_points.shape[0], noisy_points.shape[1], 
                                       self.scene_time_encoder.d_output, 
                                       device=noisy_points.device, dtype=noisy_points.dtype)
            x = torch.cat([noisy_points, diff_time_emb, zero_scene_emb], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        predicted_noise = self.fc4(x)
        
        return predicted_noise

if __name__ == '__main__':
    from configs.default_config import config
    
    model = DiffusionModule(config).cuda()
    
    points = torch.randn(2, 4096, 3).cuda()
    diffusion_timesteps = torch.randint(0, config['model']['diffusion']['time_steps'], (2,)).cuda()
    scene_timesteps = torch.rand(2, 1).cuda()
    
    output = model(points, diffusion_timesteps, scene_timesteps)
    print(f"Time-aware Diffusion Module output shape: {output.shape}")
    
    
    output_static = model(points, diffusion_timesteps)
    print(f"Static Diffusion Module output shape: {output_static.shape}") 