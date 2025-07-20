import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import logging
from tqdm import tqdm

from src.data.dataset import KITTIDataset
from src.models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF
from src.model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from src.utils.rays import get_rays, ndc_rays
from .renderer import Renderer, sample_along_rays, volume_render


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0
        
        # Setup logger
        self._setup_logger()

        # Initialize Dataset and DataLoader
        self.train_dataset = KITTIDataset(config, split='train')
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
        
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=config['training']['scheduler_step_size'], 
                                                   gamma=config['training']['scheduler_gamma'])
        
        # Loss function
        self.nerf_loss = torch.nn.MSELoss()
        self.diffusion_loss = torch.nn.MSELoss()
        self.displacement_loss = lambda x: torch.mean(x**2) # L2 regularization on displacement
        
        # Logging and Checkpoints
        self.writer = SummaryWriter(log_dir=config['output_path'])
        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load checkpoint if exists
        self._load_checkpoint()

    def _setup_logger(self):
        """Initializes the logger for training."""
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Log to file
        log_file = os.path.join(self.config['output_path'], 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Log to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    def _load_checkpoint(self):
        """Loads the latest checkpoint from the checkpoint directory."""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch_*.pth"))
        if not checkpoint_files:
            self.logger.info("No checkpoint found. Starting training from scratch.")
            return

        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        self.logger.info(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        self.logger.info(f"Resuming training from epoch {self.start_epoch}")

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.train_one_epoch(epoch)
            self.scheduler.step()
            
            if (epoch + 1) % self.config['training'].get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch)
        
        self.writer.close()
        self.logger.info("Training finished.")

    def train_one_epoch(self, epoch):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for i, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device) # (B, 3, H, W)
            lidar_points = batch['lidar_points'].to(self.device) # (B, N, 3)
            scene_timestep = batch['scene_timestep'].to(self.device)
            c2w = batch['camera_to_world'].to(self.device) # Use real camera pose
            
            H, W = images.shape[2], images.shape[3]
            # Get focal length from KITTI calibration data (handle batch properly)
            focal = batch['calibration']['P2'][0, 0, 0].item()  # For now, use first sample in batch
            
            rays_o, rays_d = get_rays(H, W, focal, c2w[0]) # Use first sample in batch
            rays_o, rays_d = rays_o.to(self.device), rays_d.to(self.device)
            
            # Sub-sample rays for faster training
            num_train_rays = 1024
            ray_indices = torch.randperm(rays_o.shape[0])[:num_train_rays]
            rays_o_train, rays_d_train = rays_o[ray_indices], rays_d[ray_indices]
            
            gt_pixels = images.permute(0, 2, 3, 1).reshape(-1, 3)[ray_indices] # (N_train_rays, 3)
            
            B = lidar_points.shape[0]
            diffusion_timesteps = torch.randint(0, self.config['model']['diffusion']['time_steps'], (B,)).to(self.device)
            
            # Sample points along rays
            pts, z_vals = sample_along_rays(rays_o_train, rays_d_train, self.renderer.near, self.renderer.far, self.renderer.n_samples)
            pts = pts.unsqueeze(0).expand(B, -1, -1, -1) # Add batch dim
            view_dirs = rays_d_train.unsqueeze(0).expand(B, -1, -1)
            
            # Forward pass - different for dynamic vs static models
            if self.is_dynamic:
                # Dynamic SSD-NeRF: Uses Diffusion + Deformation + NeRF
                outputs = self.model(lidar_points, view_dirs, pts, diffusion_timesteps, scene_timestep)
                
                # Enhanced volumetric rendering with noise regularization during training
                raw_output = outputs['nerf_output'].squeeze(0) # Remove batch dim for rendering
                rgb_map, depth_map, acc_map, disp_map = volume_render(
                    raw_output, z_vals, rays_d_train, 
                    raw_noise_std=0.1,  # Add noise for regularization
                    white_bkgd=False
                )
                
                # --- Dynamic Model Loss Calculation ---
                loss_nerf = self.nerf_loss(rgb_map, gt_pixels)
                
                # Diffusion loss (placeholder - should be improved with proper DDPM loss)
                gt_noise = torch.randn_like(outputs['diffusion_features'])
                loss_diffusion = self.diffusion_loss(outputs['diffusion_features'], gt_noise)

                # Displacement loss to regularize the deformation field
                displacement = outputs['displacement']
                loss_displacement = self.displacement_loss(displacement)

                total_loss = loss_nerf + 0.1 * loss_diffusion + 0.01 * loss_displacement
                
                # Log individual losses
                self.writer.add_scalar('Loss/NeRF', loss_nerf.item(), epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Loss/Diffusion', loss_diffusion.item(), epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Loss/Displacement', loss_displacement.item(), epoch * len(self.train_loader) + i)
                
            else:
                # Static SSD-NeRF: Traditional 2D→3D approach
                # Extract 2D bounding boxes from ground truth (simplified)
                # In real implementation, you'd use actual KITTI 3D object labels
                proposals_list = None  # For now, inference mode
                
                locs_2d, confs_2d, pred_3d_params = self.model(images, proposals_list)
                
                # For static model, we use a simpler loss (classification + localization)
                # This is a simplified version - real implementation needs proper 2D/3D losses
                if pred_3d_params is not None:
                    # 3D box prediction loss (if proposals were provided)
                    loss_3d = torch.mean(pred_3d_params**2)  # L2 regularization placeholder
                else:
                    loss_3d = torch.tensor(0.0, device=self.device)
                
                # 2D detection losses (simplified)
                loss_2d_loc = torch.mean(locs_2d**2)  # Placeholder
                loss_2d_conf = torch.mean(confs_2d**2)  # Placeholder
                
                total_loss = loss_2d_loc + loss_2d_conf + 0.1 * loss_3d
                
                # Log static model losses
                self.writer.add_scalar('Loss/2D_Localization', loss_2d_loc.item(), epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Loss/2D_Classification', loss_2d_conf.item(), epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Loss/3D_Prediction', loss_3d.item(), epoch * len(self.train_loader) + i)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            self.writer.add_scalar('Loss/Total', total_loss.item(), epoch * len(self.train_loader) + i)
            
            progress_bar.set_postfix(loss=total_loss.item(), lr=self.scheduler.get_last_lr()[0])

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")

# The main block from the original file is removed as it's not needed for the library file.
# To run this, one would use the scripts/run_training.py 