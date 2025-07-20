import torch
import torch.nn as nn
from .diffusion_module import DiffusionModule
from .nerf_module import NeRFModule
from .deformation_module import DeformationModule

def find_nearest_neighbor_features(pts, lidar_pts, features):
    """
    Find features for each point in pts by finding the nearest neighbor in lidar_pts.
    
    Args:
        pts (torch.Tensor): (B, N_rays, N_samples, 3) - Points to query.
        lidar_pts (torch.Tensor): (B, N_lidar, 3) - LiDAR points with features.
        features (torch.Tensor): (B, N_lidar, D_feature) - Features of LiDAR points.
        
    Returns:
        torch.Tensor: (B, N_rays, N_samples, D_feature) - Features for each query point.
    """
    B, N_rays, N_samples, _ = pts.shape
    
    pts_flat = pts.reshape(B, -1, 3) # (B, N_rays*N_samples, 3)
    
    # Calculate pairwise distances
    dist = torch.cdist(pts_flat, lidar_pts) # (B, N_rays*N_samples, N_lidar)
    
    # Find nearest neighbor indices
    _, nn_idx = torch.min(dist, dim=2) # (B, N_rays*N_samples)
    
    # Gather features
    nn_idx_expanded = nn_idx.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    gathered_features = torch.gather(features, 1, nn_idx_expanded) # (B, N_rays*N_samples, D_feature)
    
    return gathered_features.reshape(B, N_rays, N_samples, -1)


class SSD_NeRF(nn.Module):
    """
    Dynamic Single-Stage-Diffusion NeRF model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_module = DiffusionModule(config)
        self.deformation_module = DeformationModule(config)
        
        self.feature_dim = config['model']['diffusion']['feature_dim']
        self.nerf_module = NeRFModule(config, condition_dim=self.feature_dim)

    def forward(self, lidar_points, view_dirs, rays, diffusion_timesteps, scene_timestep):
        """
        Main forward pass for the dynamic SSD-NeRF model.
        Args:
            lidar_points (torch.Tensor): Sparse LiDAR input. (B, N_points, 3)
            view_dirs (torch.Tensor): Viewing directions for each ray. (B, N_rays, 3)
            rays (torch.Tensor): Ray samples (positions) for rendering. (B, N_rays, N_samples, 3)
            diffusion_timesteps (torch.Tensor): Timesteps for the diffusion process. (B,)
            scene_timestep (torch.Tensor): Timestep for the dynamic scene. (B, 1)
        """
        # 1. Deform observed points to canonical space
        B, N_rays, N_samples, _ = rays.shape
        rays_flat = rays.reshape(B, -1, 3)
        displacement = self.deformation_module(rays_flat, scene_timestep)
        canonical_rays_flat = rays_flat + displacement
        canonical_rays = canonical_rays_flat.reshape(B, N_rays, N_samples, 3)
        
        # 2. Get features from diffusion module conditioned on scene time
        # This requires the diffusion model to also be time-aware.
        # We'll pass the scene_timestep to the diffusion_module as well.
        diffusion_features = self.diffusion_module(lidar_points, diffusion_timesteps, scene_timestep)
        
        # 3. Sample features for each point on the *canonical* rays
        sampled_features = find_nearest_neighbor_features(canonical_rays, lidar_points, diffusion_features)
        
        # 4. Render with conditioned NeRF in canonical space
        nerf_output = self.nerf_module(canonical_rays, view_dirs, condition=sampled_features)
        
        return {
            'nerf_output': nerf_output,
            'diffusion_features': diffusion_features,
            'displacement': displacement
        }

# # __main__ block needs to be updated to test the new dynamic model
# if __name__ == '__main__':
#     from configs.default_config import config
    
#     config['model']['nerf']['condition_dim'] = config['model']['diffusion']['feature_dim']
    
#     model = SSD_NeRF(config).cuda()
    
#     batch_size = 2
#     num_points = 1024
#     num_rays = 512
#     num_samples_per_ray = 64
    
#     lidar = torch.randn(batch_size, num_points, 3).cuda()
#     rays = torch.randn(batch_size, num_rays, num_samples_per_ray, 3).cuda()
#     view_dirs = torch.randn(batch_size, num_rays, 3).cuda()
#     diffusion_t = torch.randint(0, config['model']['diffusion']['time_steps'], (batch_size,)).cuda()
#     scene_t = torch.rand(batch_size, 1).cuda()
    
#     output = model(lidar, view_dirs, rays, diffusion_t, scene_t)
    
#     print("--- Dynamic SSD-NeRF Test ---")
#     print(f"NeRF output shape: {output['nerf_output'].shape}")
#     print(f"Displacement shape: {output['displacement'].shape}") 