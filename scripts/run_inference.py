import argparse
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
import imageio

# Add the project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF
from src.model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from src.data.dataset import KITTIDataset
from src.utils.config_utils import load_config
from src.utils.ray_utils import get_rays
from src.training.renderer import Renderer, sample_along_rays, volume_render

def run_inference(config, checkpoint_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model based on configuration
    model_type = config['model'].get('type', 'dynamic')
    if model_type == 'dynamic':
        print("🚀 Loading Dynamic SSD-NeRF for inference")
        model = DynamicSSDNeRF(config).to(device)
        is_dynamic = True
    elif model_type == 'static':
        print("📷 Loading Static SSD-NeRF for inference")
        model = StaticSSDNeRF(num_classes=config['model']['ssd_nerf']['num_classes']).to(device)
        is_dynamic = False
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    dataset = KITTIDataset(config, split='test')
    renderer = Renderer(config)
    
    print(f"Running inference on {len(dataset)} samples using {model_type} model...")
    
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataset)):
            if is_dynamic:
                # Dynamic model inference
                image = sample['image'].unsqueeze(0).to(device)
                lidar_points = sample['lidar_points'].unsqueeze(0).to(device)
                scene_timestep = sample['scene_timestep'].unsqueeze(0).to(device)
                focal = sample['calibration']['P2'][0, 0].item()
                
                H, W = image.shape[2], image.shape[3]
                c2w = torch.eye(4).to(device)
                
                rays_o, rays_d = get_rays(H, W, focal, c2w)
                rays_o, rays_d = rays_o.to(device), rays_d.to(device)
                
                # Render in chunks for memory efficiency
                chunk_size = 1024
                rendered_image = []
                
                for i in range(0, rays_o.shape[0], chunk_size):
                    rays_o_chunk = rays_o[i:i+chunk_size]
                    rays_d_chunk = rays_d[i:i+chunk_size]
                    
                    pts, z_vals = sample_along_rays(rays_o_chunk, rays_d_chunk, renderer.near, renderer.far, renderer.n_samples)
                    pts_b = pts.unsqueeze(0)
                    view_dirs = rays_d_chunk.unsqueeze(0)
                    
                    timesteps = torch.randint(0, config['model']['diffusion']['time_steps'], (1,)).to(device)
                    
                    output = model(lidar_points, view_dirs, pts_b, timesteps, scene_timestep)
                    raw = output['nerf_output'].squeeze(0)
                    
                    rgb_map, depth_map, acc_map, disp_map = volume_render(raw, z_vals, rays_d[i:i+chunk_size])
                    rendered_image.append(rgb_map)
                
            else:
                # Static model inference  
                image = sample['image'].unsqueeze(0).to(device)
                
                # Static SSD-NeRF inference (2D detection only)
                locs_2d, confs_2d, pred_3d_params = model(image, None)
                
                # For visualization, just use the original image with detection overlays
                # In a real implementation, you'd draw bounding boxes based on locs_2d and confs_2d
                img_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                frames.append(img_np)
                continue
            
            # Process dynamic model output
            img = torch.cat(rendered_image, 0).reshape(H, W, 3)
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            frames.append(img_np)
            
            # Save individual frame
            frame_path = os.path.join(output_dir, f'frame_{idx:06d}.png')
            cv2.imwrite(frame_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            if idx >= 50:  # Limit for demo
                break
    
    # Create video
    if frames:
        video_path = os.path.join(output_dir, f'{model_type}_inference_result.mp4')
        imageio.mimsave(video_path, frames, fps=10)
        print(f"✅ Inference complete! Video saved to: {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Run SSD-NeRF Inference")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--output', type=str, default='output/inference', help="Output directory")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_inference(config, args.checkpoint, args.output)

if __name__ == '__main__':
    main() 