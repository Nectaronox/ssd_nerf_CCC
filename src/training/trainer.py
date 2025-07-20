import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import logging
from tqdm import tqdm
import traceback

from src.data.dataset import KITTIDataset
from src.models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF
from src.model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from src.utils.rays import get_rays, sample_points_on_rays, ndc_rays
from .renderer import Renderer, volume_render


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0
        
        # Setup logger first
        self._setup_logger()
        self.logger.info(f"🔧 트레이너 초기화 중... (Device: {self.device})")

        try:
            # Initialize Dataset and DataLoader
            self.train_dataset = KITTIDataset(config, split='train')
            self.train_loader = DataLoader(
                self.train_dataset, 
                batch_size=config['data']['batch_size'], 
                shuffle=True, 
                num_workers=config['data']['num_workers'],
                pin_memory=True if torch.cuda.is_available() else False
            )
            self.logger.info(f"📊 데이터셋 로드 완료: {len(self.train_dataset)}개 샘플")
            
            # Initialize Model based on configuration
            model_type = config['model'].get('type', 'dynamic')
            if model_type == 'dynamic':
                self.logger.info("🚀 Using Dynamic SSD-NeRF (with Diffusion + Deformation)")
                self.model = DynamicSSDNeRF(config).to(self.device)
                self.is_dynamic = True
            elif model_type == 'static':
                self.logger.info("📷 Using Static SSD-NeRF (traditional approach)")
                self.model = StaticSSDNeRF(num_classes=config['model']['ssd_nerf']['num_classes']).to(self.device)
                self.is_dynamic = False
            else:
                raise ValueError(f"Unknown model type: {model_type}. Choose 'dynamic' or 'static'")
            
            # Initialize Renderer
            self.renderer = Renderer(config)
            
            # Initialize Optimizer and Scheduler
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config['training'].get('scheduler_step_size', 30), 
                gamma=config['training'].get('scheduler_gamma', 0.1)
            )
            
            # Loss functions
            self.nerf_loss = torch.nn.MSELoss()
            self.diffusion_loss = torch.nn.MSELoss()
            self.displacement_loss = lambda x: torch.mean(x**2)  # L2 regularization on displacement
            
            # Logging and Checkpoints
            log_dir = os.path.join(config['output_path'], 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            
            self.checkpoint_dir = config['training']['checkpoint_dir']
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
            # Load checkpoint if exists
            self._load_checkpoint()
            
        except Exception as e:
            self.logger.error(f"❌ 트레이너 초기화 실패: {e}")
            raise

    def _setup_logger(self):
        self.logger = logging.getLogger("Trainer")
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        os.makedirs(self.config['output_path'], exist_ok=True)
        log_file = os.path.join(self.config['output_path'], 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def _load_checkpoint(self):
        """Loads the latest checkpoint from the checkpoint directory."""
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch_*.pth"))
            if not checkpoint_files:
                self.logger.info("No checkpoint found. Starting training from scratch.")
                return

            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            self.logger.info(f"Loading checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            
            self.logger.info(f"✅ 체크포인트 로드 완료. Epoch {self.start_epoch}부터 재시작")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 체크포인트 로드 실패: {e}. 처음부터 시작합니다.")
            self.start_epoch = 0

    def train(self):
        """Main training loop."""
        try:
            self.logger.info("🚀 학습 시작!")
            total_epochs = self.config['training']['epochs']
            
            for epoch in range(self.start_epoch, total_epochs):
                try:
                    epoch_loss = self.train_one_epoch(epoch)
                    self.scheduler.step()
                    
                    # 체크포인트 저장
                    checkpoint_interval = self.config['training'].get('checkpoint_interval', 10)
                    if (epoch + 1) % checkpoint_interval == 0:
                        self.save_checkpoint(epoch)
                    
                    self.logger.info(f"✅ Epoch {epoch+1}/{total_epochs} 완료 (Loss: {epoch_loss:.6f})")
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 사용자 중단 요청. 체크포인트 저장 중...")
                    self.save_checkpoint(epoch)
                    break
                    
                except Exception as e:
                    self.logger.error(f"❌ Epoch {epoch+1} 실행 중 오류: {e}")
                    self.logger.error(traceback.format_exc())
                    continue
            
            self.writer.close()
            self.logger.info("🎉 학습 완료!")
            
        except Exception as e:
            self.logger.error(f"❌ 학습 중 치명적 오류: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}"
        )
        
        for i, batch in enumerate(progress_bar):
            try:
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)  # (B, 3, H, W)
                lidar_points = batch['lidar_points'].to(self.device, non_blocking=True)  # (B, N, 3)
                scene_timestep = batch['scene_timestep'].to(self.device, non_blocking=True)
                c2w = batch['camera_to_world'].to(self.device, non_blocking=True)
                
                H, W = images.shape[2], images.shape[3]
                focal = batch['focal'][0].item()
                
                # Process camera-to-world matrix for rays.py compatibility  
                c2w_processed = self._process_c2w_for_rays(c2w[0])
                
                # ✅ rays.py의 get_rays 사용 (device 호환성 보장)
                rays_o, rays_d = get_rays(H, W, focal, c2w_processed)
                
                # 🔍 차원 검증 로그 (디버깅용)
                if i == 0:  # 첫 번째 배치에서만 로그
                    self.logger.debug(f"📏 Image shape: {images.shape} (B={images.shape[0]}, C={images.shape[1]}, H={H}, W={W})")
                    self.logger.debug(f"📏 rays_o shape: {rays_o.shape} (expected: ({H*W}, 3) = ({H*W}, 3))")
                    self.logger.debug(f"📏 rays_d shape: {rays_d.shape}")
                
                # Sub-sample rays for efficiency
                num_train_rays = min(self.config['training'].get('num_train_rays', 1024), rays_o.shape[0])
                ray_indices = torch.randperm(rays_o.shape[0], device=self.device)[:num_train_rays]
                rays_o_train, rays_d_train = rays_o[ray_indices], rays_d[ray_indices]
                
                # ✅ gt_pixels 차원 검증
                gt_pixels = images.permute(0, 2, 3, 1).reshape(-1, 3)[ray_indices]
                
                # 🔍 차원 일치성 검증
                if rays_o_train.shape[0] != gt_pixels.shape[0]:
                    raise ValueError(f"❌ Ray와 GT pixel 수가 맞지 않음: rays={rays_o_train.shape[0]}, pixels={gt_pixels.shape[0]}")
                
                if i == 0:  # 첫 번째 배치에서만 로그
                    self.logger.debug(f"📏 rays_o_train shape: {rays_o_train.shape}")
                    self.logger.debug(f"📏 gt_pixels shape: {gt_pixels.shape}")
                    self.logger.debug(f"📏 num_train_rays: {num_train_rays}")
                
                # Forward pass
                if self.is_dynamic:
                    loss = self._forward_dynamic_model(
                        images, lidar_points, scene_timestep, 
                        rays_o_train, rays_d_train, gt_pixels, epoch, i
                    )
                else:
                    loss = self._forward_static_model(
                        images, lidar_points, epoch, i
                    )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # ✅ 설정 기반 Gradient clipping for stability
                max_norm = self.config['training'].get('gradient_clip_max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
                
                self.optimizer.step()
                
                # Logging
                total_loss += loss.item()
                num_batches += 1
                
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('Loss/Total', loss.item(), step)
                
                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1e6:.0f}MB' if torch.cuda.is_available() else 'CPU'
                })
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"❌ Batch {i} 처리 중 오류: {e}")
                continue
        
        return total_loss / max(num_batches, 1)

    def _process_c2w_for_rays(self, c2w):
        """
        Process camera-to-world matrix for rays.py compatibility.
        rays.py expects 3x4 matrix, not 4x4.
        """
        if c2w.shape == (4, 4):
            return c2w[:3, :4]  # Convert 4x4 to 3x4
        elif c2w.shape == (3, 4):
            return c2w  # Already correct format
        else:
            self.logger.warning(f"⚠️ 예상하지 못한 c2w 행렬 크기: {c2w.shape}")
            # Force to 3x4 format
            if c2w.numel() >= 12:
                return c2w.view(3, 4)
            else:
                # Fallback to identity
                return torch.eye(3, 4, device=c2w.device, dtype=c2w.dtype)

    def _process_c2w_matrix(self, c2w):
        """Process camera-to-world matrix to ensure 4x4 format."""
        if c2w.shape[0] == 3:  # 3x4 matrix
            c2w_4x4 = torch.eye(4, device=self.device, dtype=c2w.dtype)
            c2w_4x4[:3, :4] = c2w
            return c2w_4x4
        else:  # Already 4x4
            return c2w

    def _forward_dynamic_model(self, images, lidar_points, scene_timestep, 
                              rays_o_train, rays_d_train, gt_pixels, epoch, batch_idx):
        """Forward pass for dynamic SSD-NeRF model."""
        B = lidar_points.shape[0]
        
        # Generate diffusion timesteps
        diffusion_timesteps = torch.randint(
            0, self.config['model']['diffusion']['time_steps'], (B,), device=self.device
        )
        
        # ✅ rays.py의 고급 샘플링 기능 사용
        pts, z_vals = sample_points_on_rays(
            rays_o_train, rays_d_train,
            near=self.renderer.near,
            far=self.renderer.far,
            n_samples=self.renderer.n_samples,
            perturb=self.config.get('rendering', {}).get('perturb_rays', True),  # config에서 설정 가져오기
            l_disp=self.config['training'].get('use_disparity_sampling', True)  # disparity space 샘플링
        )
        
        # NDC 변환 옵션 (필요시 활성화)
        use_ndc = self.config['training'].get('use_ndc', False)
        if use_ndc:
            H, W = images.shape[2], images.shape[3]
            focal = self.config['data'].get('focal_length', 400.0)  # config에서 focal length 가져오기
            rays_o_ndc, rays_d_ndc = ndc_rays(H, W, focal, self.renderer.near, rays_o_train, rays_d_train)
            rays_o_train, rays_d_train = rays_o_ndc, rays_d_ndc
            self.logger.debug("📐 NDC 변환 적용됨")
        
        # Process point dimensions
        pts_batch = self._process_pts_dimensions(pts, rays_o_train.shape[0], B)
        view_dirs = rays_d_train.unsqueeze(0).expand(B, -1, -1)
        
        # Model forward pass
        outputs = self.model(lidar_points, view_dirs, pts_batch, diffusion_timesteps, scene_timestep)
        
        # Validate outputs
        if not self._validate_model_outputs(outputs, required_keys=['nerf_output']):
            raise ValueError("Model outputs are missing required keys")
        
        # Volumetric rendering - config에서 파라미터 가져오기
        raw_output = outputs['nerf_output'].squeeze(0)
        rendering_config = self.config.get('rendering', {})
        rgb_map, depth_map, acc_map, disp_map = volume_render(
            raw_output, z_vals, rays_d_train, 
            raw_noise_std=rendering_config.get('raw_noise_std', 0.1),
            white_bkgd=rendering_config.get('white_background', False)
        )
        
        # ✅ 설정 기반 loss 가중치 적용
        loss_weights = self.config['training'].get('loss_weights', {
            'nerf': 1.0, 'diffusion': 0.1, 'displacement': 0.01
        })
        
        # Calculate losses
        loss_nerf = self.nerf_loss(rgb_map, gt_pixels)
        
        # Diffusion loss
        if 'diffusion_features' in outputs:
            gt_noise = torch.randn_like(outputs['diffusion_features'])
            loss_diffusion = self.diffusion_loss(outputs['diffusion_features'], gt_noise)
        else:
            loss_diffusion = torch.tensor(0.0, device=self.device)
        
        # Displacement loss
        if 'displacement' in outputs:
            loss_displacement = self.displacement_loss(outputs['displacement'])
        else:
            loss_displacement = torch.tensor(0.0, device=self.device)
        
        # ✅ 설정 기반 가중치 적용
        total_loss = (loss_weights['nerf'] * loss_nerf + 
                     loss_weights['diffusion'] * loss_diffusion + 
                     loss_weights['displacement'] * loss_displacement)
        
        # Log individual losses
        step = epoch * len(self.train_loader) + batch_idx
        self.writer.add_scalar('Loss/NeRF', loss_nerf.item(), step)
        self.writer.add_scalar('Loss/Diffusion', loss_diffusion.item(), step)
        self.writer.add_scalar('Loss/Displacement', loss_displacement.item(), step)
        self.writer.add_scalar('Loss/Weighted_Total', total_loss.item(), step)
        
        # 추가 메트릭 로깅
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Metrics/Depth_Mean', depth_map.mean().item(), step)
            self.writer.add_scalar('Metrics/Acc_Mean', acc_map.mean().item(), step)
            # RGB 색상 분포 모니터링
            self.writer.add_scalar('Metrics/RGB_Mean', rgb_map.mean().item(), step)
            self.writer.add_scalar('Metrics/RGB_Std', rgb_map.std().item(), step)
        
        return total_loss

    def _forward_static_model(self, images, lidar_points, epoch, batch_idx):
        """Forward pass for static SSD-NeRF model."""
        proposals_list = None  # For inference mode
        
        locs_2d, confs_2d, pred_3d_params = self.model(images, proposals_list)
        
        # Calculate losses
        if pred_3d_params is not None:
            loss_3d = torch.mean(pred_3d_params**2)
        else:
            loss_3d = torch.tensor(0.0, device=self.device)
        
        loss_2d_loc = torch.mean(locs_2d**2)
        loss_2d_conf = torch.mean(confs_2d**2)
        
        total_loss = loss_2d_loc + loss_2d_conf + 0.1 * loss_3d
        
        # Log losses
        step = epoch * len(self.train_loader) + batch_idx
        self.writer.add_scalar('Loss/2D_Localization', loss_2d_loc.item(), step)
        self.writer.add_scalar('Loss/2D_Classification', loss_2d_conf.item(), step)
        self.writer.add_scalar('Loss/3D_Prediction', loss_3d.item(), step)
        
        return total_loss

    def _process_pts_dimensions(self, pts, num_rays, batch_size):
        """Process pts tensor dimensions to (B, N_rays, N_samples, 3)."""
        if pts.dim() == 4:  # (H, W, N_samples, 3)
            H, W = pts.shape[0], pts.shape[1]
            if H * W > num_rays:
                # Reshape and subsample
                pts_reshaped = pts.reshape(H * W, self.renderer.n_samples, 3)
                pts_subsampled = pts_reshaped[:num_rays]  # Take first num_rays
                return pts_subsampled.unsqueeze(0).expand(batch_size, -1, -1, -1)
            else:
                return pts.reshape(1, -1, self.renderer.n_samples, 3).expand(batch_size, -1, -1, -1)
        elif pts.dim() == 3:  # (N_rays, N_samples, 3)
            return pts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            # Force reshape
            return pts.reshape(batch_size, num_rays, self.renderer.n_samples, 3)

    def _validate_model_outputs(self, outputs, required_keys):
        """Validate that model outputs contain required keys."""
        if not isinstance(outputs, dict):
            return False
        
        for key in required_keys:
            if key not in outputs:
                self.logger.warning(f"⚠️ 모델 출력에 '{key}' 키가 없습니다.")
                return False
        return True

    def save_checkpoint(self, epoch):
        """Save training checkpoint."""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
            
            # Keep only last 3 checkpoints to save space
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 저장 실패: {e}")

    def _cleanup_old_checkpoints(self, keep_last=3):
        """Remove old checkpoints to save disk space."""
        try:
            checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch_*.pth"))
            if len(checkpoint_files) > keep_last:
                # Sort by modification time
                checkpoint_files.sort(key=os.path.getmtime)
                # Remove oldest files
                for old_checkpoint in checkpoint_files[:-keep_last]:
                    os.remove(old_checkpoint)
                    self.logger.info(f"🗑️ 오래된 체크포인트 삭제: {old_checkpoint}")
        except Exception as e:
            self.logger.warning(f"⚠️ 체크포인트 정리 실패: {e}")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'writer') and self.writer:
            self.writer.close() 