import torch

def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """
    Generate rays from camera coordinates.

    Args:
        H (int): Image height in pixels.
        W (int): Image width in pixels.
        focal (float): Focal length of the camera.
        c2w (torch.Tensor): Camera-to-world transformation matrix (3x4).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rays_o (torch.Tensor): Origins of the rays (H*W, 3).
            - rays_d (torch.Tensor): Directions of the rays (H*W, 3).
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=c2w.device),
        torch.arange(H, dtype=torch.float32, device=c2w.device),
        indexing='xy'
    )
    
    dirs = torch.stack([
        (i - W * .5) / focal,
        -(j - H * .5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    
    rays_d = torch.sum(
        dirs[..., None, :] * c2w[:3, :3], dim=-1
    )
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    # ✅ 차원 수정: (H, W, 3) -> (H*W, 3)으로 reshape하여 doc string과 일치시킴
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def sample_points_on_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    n_samples: int,
    perturb: bool = True,
    l_disp: bool = True,
):
    """
    Sample points along rays.

    Args:
        rays_o (torch.Tensor): Origins of the rays.
        rays_d (torch.Tensor): Directions of the rays.
        near (float): Near bound for sampling.
        far (float): Far bound for sampling.
        n_samples (int): Number of samples per ray.
        perturb (bool): If True, apply random perturbation to sample points.
        l_disp (bool): If True, sample linearly in disparity space.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - pts (torch.Tensor): Sampled points in 3D space.
            - z_vals (torch.Tensor): Depth values of the sampled points.
    """
    n_rays = rays_o.shape[0]
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    
    if l_disp:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        z_vals = near * (1. - t_vals) + far * t_vals
        
    z_vals = z_vals.expand([n_rays, n_samples])
    
    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand
        
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

def ndc_rays(H: int, W: int, focal: float, near: float, rays_o: torch.Tensor, rays_d: torch.Tensor):
    """
    Convert rays to Normalized Device Coordinates (NDC).

    Args:
        H (int): Height of the image.
        W (int): Width of the image.
        focal (float): Focal length of the camera.
        near (float): Near bound.
        rays_o (torch.Tensor): Ray origins.
        rays_d (torch.Tensor): Ray directions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rays_o_ndc (torch.Tensor): Ray origins in NDC space.
            - rays_d_ndc (torch.Tensor): Ray directions in NDC space.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o_ndc = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o_ndc[..., 0] / rays_o_ndc[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o_ndc[..., 1] / rays_o_ndc[..., 2]
    o2 = 1. + 2. * near / rays_o_ndc[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o_ndc[..., 0] / rays_o_ndc[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o_ndc[..., 1] / rays_o_ndc[..., 2])
    d2 = -2. * near / rays_o_ndc[..., 2]

    rays_o_ndc = torch.stack([o0, o1, o2], -1)
    rays_d_ndc = torch.stack([d0, d1, d2], -1)

    return rays_o_ndc, rays_d_ndc 