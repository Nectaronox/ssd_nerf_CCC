"""
Enhanced Predictor for SSD-NeRF Models
Supports both Dynamic and Static SSD-NeRF variants for real-time inference
"""

import torch
import numpy as np
from torchvision.ops import nms
from typing import Dict, List, Optional, Union
import time

from ..models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF
from ..model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from ..training.renderer import volume_render, sample_along_rays
from ..utils.ray_utils import get_rays

class SSDNeRFPredictor:
    """
    Real-time predictor for SSD-NeRF models.
    
    Supports both:
    - Dynamic SSD-NeRF: Full 3D scene reconstruction with temporal modeling
    - Static SSD-NeRF: Traditional 2D→3D object detection
    
    Optimized for real-time inference in autonomous driving scenarios.
    """
    
    def __init__(self, 
                 model_path: str, 
                 config: dict,
                 model_type: str = 'dynamic',
                 conf_thresh: float = 0.5, 
                 nms_thresh: float = 0.4):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            model_type: 'dynamic' or 'static'
            conf_thresh: Confidence threshold for detection
            nms_thresh: NMS threshold for detection
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model_type = model_type
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # Load appropriate model
        if model_type == 'dynamic':
            print("🚀 Loading Dynamic SSD-NeRF for real-time inference")
            self.model = DynamicSSDNeRF(config).to(self.device)
            self.is_dynamic = True
        elif model_type == 'static':
            print("📷 Loading Static SSD-NeRF for real-time inference")
            self.model = StaticSSDNeRF(num_classes=config['model']['ssd_nerf']['num_classes']).to(self.device)
            self.is_dynamic = False
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Performance monitoring
        self.inference_times = []
        
        print(f"✅ {model_type.capitalize()} SSD-NeRF predictor initialized")

    def predict_single_image(self, 
                            image: torch.Tensor,
                            lidar_points: Optional[torch.Tensor] = None,
                            scene_timestep: float = 0.0,
                            camera_intrinsics: Optional[Dict] = None) -> Dict:
        """
        Make predictions for a single image (real-time inference).
        
        Args:
            image: Input image tensor (C, H, W)
            lidar_points: LiDAR point cloud (N, 3) - required for dynamic model
            scene_timestep: Temporal information for dynamic scenes
            camera_intrinsics: Camera parameters {'focal': float, 'c2w': np.ndarray}
            
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        
        with torch.no_grad():
            if self.is_dynamic:
                result = self._predict_dynamic(image, lidar_points, scene_timestep, camera_intrinsics)
            else:
                result = self._predict_static(image)
        
        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        result.update({
            'inference_time': inference_time,
            'fps': 1.0 / inference_time,
            'model_type': self.model_type
        })
        
        return result

    def _predict_dynamic(self, 
                        image: torch.Tensor, 
                        lidar_points: torch.Tensor,
                        scene_timestep: float,
                        camera_intrinsics: Optional[Dict]) -> Dict:
        """Dynamic SSD-NeRF prediction with full 3D scene reconstruction."""
        
        if lidar_points is None:
            raise ValueError("LiDAR points required for dynamic SSD-NeRF")
        
        # Prepare inputs
        image_batch = image.unsqueeze(0).to(self.device)
        lidar_batch = lidar_points.unsqueeze(0).to(self.device)
        timestep_batch = torch.tensor([scene_timestep]).unsqueeze(0).to(self.device)
        
        # Camera parameters
        H, W = image.shape[1], image.shape[2]
        if camera_intrinsics:
            focal = camera_intrinsics['focal']
            c2w = torch.from_numpy(camera_intrinsics['c2w']).float().to(self.device)
        else:
            # Default KITTI parameters
            focal = 721.5
            c2w = torch.eye(4).to(self.device)
        
        # Generate rays (subsampled for real-time performance)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o, rays_d = rays_o.to(self.device), rays_d.to(self.device)
        
        # Subsample rays for real-time performance
        num_rays = min(4096, rays_o.shape[0])  # Limit for speed
        ray_indices = torch.randperm(rays_o.shape[0])[:num_rays]
        rays_o_sub = rays_o[ray_indices]
        rays_d_sub = rays_d[ray_indices]
        
        # Sample points along rays
        pts, z_vals = sample_along_rays(rays_o_sub, rays_d_sub, 0.5, 50.0, 64)
        pts_batch = pts.unsqueeze(0)
        view_dirs_batch = rays_d_sub.unsqueeze(0)
        
        # Diffusion timesteps
        diffusion_timesteps = torch.randint(0, self.config['model']['diffusion']['time_steps'], (1,)).to(self.device)
        
        # Model forward pass
        outputs = self.model(lidar_batch, view_dirs_batch, pts_batch, diffusion_timesteps, timestep_batch)
        
        # Render 3D scene
        raw_output = outputs['nerf_output'].squeeze(0)
        rgb_map, depth_map, acc_map, disp_map = volume_render(raw_output, z_vals, rays_d_sub)
        
        # Extract dynamic objects from displacement field
        displacement = outputs['displacement']
        moving_objects = self._extract_dynamic_objects(displacement, pts_batch, threshold=0.1)
        
        # Reconstruct full image (interpolate subsampled results)
        full_rgb = self._interpolate_to_full_image(rgb_map, ray_indices, H, W)
        full_depth = self._interpolate_to_full_image(depth_map, ray_indices, H, W)
        
        return {
            'detections': moving_objects,
            'rendered_rgb': full_rgb.cpu().numpy(),
            'depth_map': full_depth.cpu().numpy(),
            'disparity_map': disp_map.cpu().numpy(),
            'environment_features': outputs['diffusion_features'].cpu().numpy(),
            'displacement_field': displacement.cpu().numpy(),
            'scene_understanding': self._analyze_scene_dynamics(outputs)
        }

    def _predict_static(self, image: torch.Tensor) -> Dict:
        """Static SSD-NeRF prediction for 2D→3D object detection."""
        
        image_batch = image.unsqueeze(0).to(self.device)
        
        # Forward pass
        locs_2d, confs_2d, pred_3d_params = self.model(image_batch, None)
        
        # Post-process 2D detections
        detections_2d = self._postprocess_2d_detections(locs_2d, confs_2d)
        
        # Extract 3D predictions if available
        if pred_3d_params is not None:
            detections_3d = self._postprocess_3d_predictions(pred_3d_params, detections_2d)
        else:
            detections_3d = []
        
        return {
            'detections': detections_3d,
            'detections_2d': detections_2d,
            'rendered_rgb': image.permute(1, 2, 0).cpu().numpy(),
            'depth_map': None,
            'disparity_map': None,
            'environment_features': None,
            'displacement_field': None,
            'scene_understanding': {'static_objects': len(detections_3d)}
        }

    def _extract_dynamic_objects(self, displacement: torch.Tensor, pts: torch.Tensor, threshold: float) -> List[Dict]:
        """Extract moving objects from displacement field."""
        
        displacement_magnitude = torch.norm(displacement, dim=-1)
        moving_mask = displacement_magnitude > threshold
        
        if moving_mask.sum() == 0:
            return []
        
        moving_points = pts.squeeze(0)[moving_mask.squeeze(0)]
        
        # Simple clustering (in production, use more sophisticated methods)
        objects = []
        if moving_points.shape[0] > 10:  # Minimum points for object
            min_coords = torch.min(moving_points, dim=0)[0]
            max_coords = torch.max(moving_points, dim=0)[0]
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            
            # Estimate object class based on size
            volume = torch.prod(size).item()
            if volume > 8.0:  # Large object
                obj_class = 'Vehicle'
                confidence = 0.8
            elif volume > 2.0:  # Medium object
                obj_class = 'Pedestrian'
                confidence = 0.7
            else:  # Small object
                obj_class = 'Cyclist'
                confidence = 0.6
            
            objects.append({
                'bbox_3d': torch.cat([center, size, torch.tensor([0.0], device=self.device)]).cpu().numpy(),
                'class': obj_class,
                'confidence': confidence,
                'velocity': torch.norm(displacement[moving_mask].mean(dim=0)).item(),
                'track_id': len(self.inference_times)  # Simple ID
            })
        
        return objects

    def _postprocess_2d_detections(self, locs_2d: torch.Tensor, confs_2d: torch.Tensor) -> List[Dict]:
        """Post-process 2D detections with NMS."""
        
        # Simplified 2D detection post-processing
        scores = torch.softmax(confs_2d, dim=-1)
        best_scores, best_classes = scores.max(dim=-1)
        
        # Filter by confidence
        keep = best_scores > self.conf_thresh
        
        if keep.sum() == 0:
            return []
        
        filtered_boxes = locs_2d[keep]
        filtered_scores = best_scores[keep]
        filtered_classes = best_classes[keep]
        
        # Apply NMS
        if filtered_boxes.shape[0] > 1:
            keep_nms = nms(filtered_boxes, filtered_scores, self.nms_thresh)
            final_boxes = filtered_boxes[keep_nms]
            final_scores = filtered_scores[keep_nms]
            final_classes = filtered_classes[keep_nms]
        else:
            final_boxes = filtered_boxes
            final_scores = filtered_scores
            final_classes = filtered_classes
        
        # Format results
        detections = []
        class_names = ['Car', 'Van', 'Truck']
        
        for i in range(final_boxes.shape[0]):
            detections.append({
                'bbox_2d': final_boxes[i].cpu().numpy(),
                'class': class_names[final_classes[i].item()],
                'confidence': final_scores[i].item()
            })
        
        return detections

    def _postprocess_3d_predictions(self, pred_3d: torch.Tensor, detections_2d: List[Dict]) -> List[Dict]:
        """Combine 3D predictions with 2D detections."""
        
        detections_3d = []
        pred_3d_np = pred_3d.cpu().numpy()
        
        for i, det_2d in enumerate(detections_2d):
            if i < pred_3d_np.shape[1]:  # Check bounds
                det_3d = det_2d.copy()
                det_3d['bbox_3d'] = pred_3d_np[0, i, :7]  # [x, y, z, l, w, h, yaw]
                det_3d['track_id'] = i
                detections_3d.append(det_3d)
        
        return detections_3d

    def _interpolate_to_full_image(self, values: torch.Tensor, indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Interpolate subsampled ray results to full image."""
        
        # Create full image tensor
        if values.dim() == 1:
            full_image = torch.zeros(H * W, device=values.device)
        else:
            full_image = torch.zeros(H * W, values.shape[-1], device=values.device)
        
        # Place values at sampled positions
        full_image[indices] = values
        
        # Simple interpolation (in production, use more sophisticated methods)
        full_image = full_image.reshape(H, W, -1) if values.dim() > 1 else full_image.reshape(H, W)
        
        return full_image

    def _analyze_scene_dynamics(self, outputs: Dict) -> Dict:
        """Analyze scene dynamics from model outputs."""
        
        displacement = outputs['displacement']
        features = outputs['diffusion_features']
        
        # Compute scene-level statistics
        displacement_stats = {
            'max_displacement': torch.max(torch.norm(displacement, dim=-1)).item(),
            'mean_displacement': torch.mean(torch.norm(displacement, dim=-1)).item(),
            'dynamic_ratio': (torch.norm(displacement, dim=-1) > 0.05).float().mean().item()
        }
        
        feature_stats = {
            'feature_diversity': torch.std(features).item(),
            'feature_magnitude': torch.norm(features).item()
        }
        
        return {
            'displacement_analysis': displacement_stats,
            'feature_analysis': feature_stats,
            'scene_complexity': displacement_stats['dynamic_ratio'] * feature_stats['feature_diversity']
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        return {
            'avg_inference_time': float(np.mean(times)),
            'max_inference_time': float(np.max(times)),
            'min_inference_time': float(np.min(times)),
            'std_inference_time': float(np.std(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_predictions': len(times),
            'model_type': self.model_type
        }

    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []


# Legacy compatibility (for existing code)
class Predictor(SSDNeRFPredictor):
    """Backward compatibility wrapper."""
    
    def __init__(self, model_path, num_classes=3, conf_thresh=0.5, nms_thresh=0.4):
        # Default to static model for backward compatibility
        config = {
            'model': {
                'ssd_nerf': {'num_classes': num_classes}
            }
        }
        super().__init__(model_path, config, 'static', conf_thresh, nms_thresh)
    
    def predict(self, image):
        """Legacy predict method."""
        result = self.predict_single_image(image)
        
        # Convert to legacy format
        predictions = []
        for det in result.get('detections', []):
            pred = {
                'box_2d': det.get('bbox_2d', np.array([0, 0, 100, 100])),
                'score': det.get('confidence', 0.5),
                'box_3d': det.get('bbox_3d', np.zeros(7))
            }
            predictions.append(pred)
        
        return predictions
