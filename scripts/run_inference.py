import argparse
import torch
import numpy as np
import cv2
import os
import logging
from tqdm import tqdm
import imageio
import traceback

# Add the project root to the Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF
from src.model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from src.data.dataset import KITTIDataset
from src.utils.config_utils import load_config
# ✅ rays.py 사용으로 변경
from src.utils.rays import get_rays, sample_points_on_rays
from src.training.renderer import Renderer, volume_render

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_inference(config, checkpoint_path, output_dir, max_frames=50, use_dummy_data=False):
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 Using device: {device}")
    
    try:
        # Load model based on configuration
        model_type = config['model'].get('type', 'dynamic')
        if model_type == 'dynamic':
            logger.info("🚀 Loading Dynamic SSD-NeRF for inference")
            model = DynamicSSDNeRF(config).to(device)
            is_dynamic = True
        elif model_type == 'static':
            logger.info("📷 Loading Static SSD-NeRF for inference")
            model = StaticSSDNeRF(num_classes=config['model']['ssd_nerf']['num_classes']).to(device)
            is_dynamic = False
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✅ Model loaded successfully")
        
        # Load dataset
        logger.info("📊 Loading KITTI dataset...")
        try:
            dataset = KITTIDataset(config, split='test', create_dummy_data=use_dummy_data)
            logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
        except FileNotFoundError as e:
            if not use_dummy_data:
                logger.error("❌ KITTI 데이터셋을 찾을 수 없습니다.")
                logger.error("💡 더미 데이터로 시도하려면 --use_dummy_data 플래그를 사용하세요.")
                raise
            else:
                logger.error(f"❌ 더미 데이터 생성에 실패했습니다: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ 데이터셋 로드 중 오류: {e}")
            if not use_dummy_data:
                logger.error("💡 더미 데이터로 시도해보세요: --use_dummy_data")
            raise
        
        renderer = Renderer(config)
        
        logger.info(f"Running inference on {min(max_frames, len(dataset))} samples using {model_type} model...")
        
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(dataset, desc="Processing frames")):
                if idx >= max_frames:
                    break
                    
                try:
                    if is_dynamic:
                        # ✅ Dynamic model inference - 개선된 처리
                        rendered_frame = _process_dynamic_frame(
                            model, sample, device, renderer, config, logger
                        )
                    else:
                        # ✅ Static model inference - 개선된 처리
                        rendered_frame = _process_static_frame(
                            model, sample, device, logger
                        )
                    
                    if rendered_frame is not None:
                        frames.append(rendered_frame)
                        
                        # Save individual frame
                        frame_path = os.path.join(output_dir, f'frame_{idx:06d}.png')
                        cv2.imwrite(frame_path, cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                    
                except Exception as e:
                    logger.error(f"❌ Error processing frame {idx}: {e}")
                    logger.error(traceback.format_exc())
                    continue
        
        # Create video
        if frames:
            video_path = os.path.join(output_dir, f'{model_type}_inference_result.mp4')
            logger.info(f"🎬 Creating video with {len(frames)} frames...")
            imageio.mimsave(video_path, frames, fps=10)
            logger.info(f"✅ Inference complete! Video saved to: {video_path}")
        else:
            logger.warning("⚠️ No frames processed successfully")
            
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise

def _process_dynamic_frame(model, sample, device, renderer, config, logger):
    """✅ 개선된 Dynamic model frame 처리"""
    try:
        # 데이터 준비
        image = sample['image'].unsqueeze(0).to(device)
        lidar_points = sample['lidar_points'].unsqueeze(0).to(device)
        scene_timestep = sample['scene_timestep'].unsqueeze(0).to(device)
        
        # ✅ 개선된 focal과 c2w 처리
        focal = sample['focal'].item()
        c2w = sample['camera_to_world']  # 이미 올바른 형태
        
        H, W = image.shape[2], image.shape[3]
        
        # ✅ c2w 차원 처리 개선
        if c2w.shape == (4, 4):
            c2w_processed = c2w[:3, :4]  # 4x4 -> 3x4
        else:
            c2w_processed = c2w  # 이미 3x4
        
        c2w_processed = c2w_processed.to(device)
        
        # ✅ rays.py 사용
        rays_o, rays_d = get_rays(H, W, focal, c2w_processed)
        
        # 메모리 효율성을 위한 ray subsampling
        chunk_size = min(2048, rays_o.shape[0])
        ray_indices = torch.randperm(rays_o.shape[0])[:chunk_size]
        rays_o_chunk = rays_o[ray_indices]
        rays_d_chunk = rays_d[ray_indices]
        
        # ✅ rays.py의 sample_points_on_rays 사용
        pts, z_vals = sample_points_on_rays(
            rays_o_chunk, rays_d_chunk,
            near=renderer.near,
            far=renderer.far,
            n_samples=renderer.n_samples,
            perturb=False,  # inference 시에는 deterministic
            l_disp=config['training'].get('use_disparity_sampling', True)
        )
        
        # 차원 맞추기
        pts_batch = pts.unsqueeze(0)  # (1, N_rays, N_samples, 3)
        view_dirs_batch = rays_d_chunk.unsqueeze(0)  # (1, N_rays, 3)
        
        # 차원 검증
        if pts_batch.shape[0] != 1:
            raise ValueError(f"Batch size mismatch: expected 1, got {pts_batch.shape[0]}")
        
        # Diffusion timesteps
        diffusion_timesteps = torch.randint(
            0, config['model']['diffusion']['time_steps'], (1,), device=device
        )
        
        # Model forward pass
        outputs = model(lidar_points, view_dirs_batch, pts_batch, diffusion_timesteps, scene_timestep)
        
        # Volume rendering
        raw_output = outputs['nerf_output'].squeeze(0)
        rgb_map, depth_map, acc_map, disp_map = volume_render(
            raw_output, z_vals, rays_d_chunk,
            raw_noise_std=0.0,  # inference 시에는 noise 없음
            white_bkgd=config.get('rendering', {}).get('white_background', False)
        )
        
        # 전체 이미지 재구성
        full_image = torch.zeros(H, W, 3, device=device)
        
        # ray_indices로 원래 위치에 매핑
        i_coords = ray_indices // W
        j_coords = ray_indices % W
        full_image[i_coords, j_coords] = rgb_map
        
        # 나머지 픽셀은 원본 이미지로 채움
        original_reshaped = image.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
        mask = torch.zeros(H, W, device=device, dtype=torch.bool)
        mask[i_coords, j_coords] = True
        full_image[~mask] = original_reshaped[~mask]
        
        # NumPy로 변환
        img_np = (torch.clamp(full_image, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        return img_np
        
    except Exception as e:
        logger.error(f"Dynamic frame processing error: {e}")
        return None

def _process_static_frame(model, sample, device, logger):
    """✅ 개선된 Static model frame 처리"""
    try:
        image = sample['image'].unsqueeze(0).to(device)
        
        # Static SSD-NeRF inference (2D detection only)
        locs_2d, confs_2d, pred_3d_params = model(image, None)
        
        # 시각화를 위해 원본 이미지 사용
        # 실제 구현에서는 locs_2d와 confs_2d를 바탕으로 bounding box를 그려야 함
        img_tensor = image.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
        
        # Denormalize (ImageNet normalization 가정)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std = torch.tensor([0.229, 0.224, 0.225], device=device)
        img_denorm = img_tensor * std + mean
        
        img_np = (torch.clamp(img_denorm, 0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        return img_np
        
    except Exception as e:
        logger.error(f"Static frame processing error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="🚀 SSD-NeRF Inference Script (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 사용법:
  python scripts/run_inference.py --config configs/default_config.py --checkpoint output/checkpoints/model_epoch_10.pth
  python scripts/run_inference.py --config configs/static_config.py --checkpoint static_model.pth --max_frames 100
        """
    )
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--output', type=str, default='output/inference', help="Output directory")
    parser.add_argument('--max_frames', type=int, default=50, help="Maximum frames to process")
    parser.add_argument('--use_dummy_data', action='store_true', help="Use dummy data for testing if KITTI dataset is not found")
    
    args = parser.parse_args()
    
    # 입력 검증
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return
    
    # 설정 로드 및 실행
    config = load_config(args.config)
    run_inference(config, args.checkpoint, args.output, args.max_frames, args.use_dummy_data)

if __name__ == '__main__':
    main() 