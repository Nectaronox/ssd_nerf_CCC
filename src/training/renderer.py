import torch
import torch.nn.functional as F

def sample_along_rays(rays_o, rays_d, near, far, n_samples, perturb=True):
    """
    Sample points along rays.
    """
    t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
    z_vals = near * (1.-t_vals) + far * t_vals

    if perturb:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=rays_o.device)
        z_vals = lower + (upper - lower) * t_rand
    
    # (N_rays, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals

def volume_render(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """
    Enhanced differentiable volume rendering with additional features.
    
    Args:
        raw (torch.Tensor): Raw output from NeRF model. (N_rays, N_samples, 4)
        z_vals (torch.Tensor): Integration time. (N_rays, N_samples)
        rays_d (torch.Tensor): Ray directions. (N_rays, 3)
        raw_noise_std (float): Standard deviation of noise added to raw densities. Default: 0
        white_bkgd (bool): If True, assume a white background. Default: False
        
    Returns:
        rgb_map (torch.Tensor): Rendered colors. (N_rays, 3)
        depth_map (torch.Tensor): Estimated depth. (N_rays,)
        acc_map (torch.Tensor): Accumulated opacity. (N_rays,)
        disp_map (torch.Tensor): Disparity map. (N_rays,)
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(rays_d.device).expand(dists[..., :1].shape)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    
    # Add noise for regularization during training
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std
    
    density = F.relu(raw[..., 3] + noise)
    
    alpha = 1. - torch.exp(-density * dists)
    
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    
    # Calculate disparity map (inverse depth)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    
    # Handle white background
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, disp_map

class Renderer:
    def __init__(self, config):
        self.n_samples = config.get('renderer', {}).get('n_samples', 64)
        self.near = config.get('renderer', {}).get('near', 0.1)
        self.far = config.get('renderer', {}).get('far', 100.0)

    def render_rays(self, model, rays_o, rays_d):
        pts, z_vals = sample_along_rays(rays_o, rays_d, self.near, self.far, self.n_samples)
        
        # Prepare inputs for NeRF model
        # The model expects view directions to be normalized.
        view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # The NeRF model in this project expects inputs in a slightly different shape.
        # It was designed for (B, N_rays, N_samples, 3). Let's adapt.
        # Assuming we process one image at a time, batch size is implicitly 1.
        B = 1 
        N_rays = rays_o.shape[0]
        
        pts_reshaped = pts.reshape(B, N_rays, self.n_samples, 3)
        view_dirs_reshaped = view_dirs.reshape(B, N_rays, 3)
        
        raw_output = model(pts_reshaped, view_dirs_reshaped)
        raw_output = raw_output.reshape(N_rays, self.n_samples, 4) # Reshape back
        
        rgb_map, depth_map, acc_map, disp_map = volume_render(raw_output, z_vals, rays_d)
        
        return {
            'rgb_map': rgb_map,
            'depth_map': depth_map,
            'acc_map': acc_map,
            'disp_map': disp_map  # Add disparity map to output
        }

if __name__ == '__main__':
    from configs.default_config import config
    from src.models.nerf_module import NeRFModule

    # Dummy NeRF model
    nerf_model = NeRFModule(config).cuda()
    renderer = Renderer(config)
    
    rays_o = torch.randn(1024, 3).cuda()
    rays_d = torch.randn(1024, 3).cuda()
    
    with torch.no_grad():
        render_output = renderer.render_rays(nerf_model, rays_o, rays_d)

    print("--- Renderer Test ---")
    print(f"Rendered RGB map shape: {render_output['rgb_map'].shape}")
    print(f"Rendered Depth map shape: {render_output['depth_map'].shape}")
    print(f"Rendered Accumulation map shape: {render_output['acc_map'].shape}") 