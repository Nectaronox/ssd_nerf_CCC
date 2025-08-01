import torch
import numpy as np

# ⚠️ DEPRECATED: 이 파일은 더 이상 권장되지 않습니다.
# 대신 src/utils/rays.py를 사용하세요. rays.py는 다음 장점을 제공합니다:
# - 타입 힌트 지원
# - Device 호환성
# - NDC 변환 기능  
# - Ray 샘플링 기능
# - 더 효율적인 구현

def get_rays(H, W, focal, c2w):
    """
    Generate rays from camera parameters.
    
    ⚠️ DEPRECATED: Use src/utils/rays.py instead for better functionality.
    
    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length of the camera.
        c2w (torch.Tensor): Camera to world transformation matrix (4x4).
    Returns:
        rays_o (torch.Tensor): Ray origins (N_rays, 3).
        rays_d (torch.Tensor): Ray directions (N_rays, 3).
    """
    print("⚠️ WARNING: ray_utils.py is deprecated. Use rays.py instead!")
    
    i, j = torch.meshgrid(torch.linspace(0.5, W - 0.5, W), torch.linspace(0.5, H - 0.5, H), indexing='xy')
    i, j = i.t(), j.t()
    
    # Directions in camera coordinates
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

if __name__ == '__main__':
    # Example usage
    H, W, focal = 400, 400, 120.0
    
    # Dummy camera-to-world matrix (identity matrix)
    c2w = torch.eye(4) 
    
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    
    print("--- Ray Generation Utility ---")
    print(f"Image size: {H}x{W}, Focal length: {focal}")
    print(f"Ray origins shape: {rays_o.shape}")
    print(f"Ray directions shape: {rays_d.shape}")
    
    # Check if directions for the center pixel are pointing straight
    center_idx = (H // 2) * W + (W // 2)
    print(f"Center ray direction: {rays_d[center_idx]}")
    # Should be close to [0, 0, -1] for identity c2w 