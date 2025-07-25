import torch
import torch.nn as nn
from ..utils.encoding import PositionalEncoder

class DeformationModule(nn.Module):
    """
    MLP that learns a deformation field from a 4D input (x, y, z, t). time encoding is used.
    """
    def __init__(self, config):
        super().__init__()
        # Use the unified PositionalEncoder from utils
        self.time_encoder = PositionalEncoder(d_input=1, n_freqs=10)
        
        # ✅ 차원 계산 명시적으로 표시
        time_dim = self.time_encoder.d_output  # 1 * (1 + 2*10) = 21
        input_dim = 3 + time_dim  # 3 for xyz, 21 for time encoding = 24
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # Output is a 3D displacement vector (dx, dy, dz)
        )

    def forward(self, pts, t):
        """
        Args:
            pts (torch.Tensor): (B, N, 3) - points in space.
            t (torch.Tensor): (B, 1) or (1, 1) - timestep for the scene.
        
        Returns:
            torch.Tensor: (B, N, 3) - displacement vectors.
        """
        # 🔍 입력 차원 검증
        B, N, spatial_dim = pts.shape
        assert spatial_dim == 3, f"Expected spatial_dim=3, got {spatial_dim}"
        
        if t.shape[0] == 1 and pts.shape[0] > 1:
            t = t.expand(pts.shape[0], -1)

        t_encoded = self.time_encoder(t).unsqueeze(1).expand(-1, pts.shape[1], -1)
        
        # Concatenate points with encoded time
        mlp_input = torch.cat([pts, t_encoded], dim=-1)
        
        # ✅ 차원 검증 추가
        expected_input_dim = 3 + self.time_encoder.d_output
        if mlp_input.shape[-1] != expected_input_dim:
            raise ValueError(f"❌ DeformationModule input 차원 불일치: expected {expected_input_dim}, got {mlp_input.shape[-1]}")
        
        # Predict displacement
        displacement = self.mlp(mlp_input)
        
        return displacement

if __name__ == '__main__':
    from configs.default_config import config
    
    model = DeformationModule(config).cuda()
    
    points = torch.randn(2, 4096, 3).cuda()
    time = torch.rand(2, 1).cuda() # Normalized time between 0 and 1
    
    displacement = model(points, time)
    
    deformed_points = points + displacement
    
    print("--- Deformation Module Test ---")
    print(f"Input points shape: {points.shape}")
    print(f"Time shape: {time.shape}")
    print(f"Displacement output shape: {displacement.shape}")
    print(f"Deformed points shape: {deformed_points.shape}") 