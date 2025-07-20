"""
Comprehensive Evaluation System for SSD-NeRF Models
Supports evaluation of both Dynamic and Static SSD-NeRF variants
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import os
from typing import Dict, List, Optional
import json
from tqdm import tqdm

# Import both model types
from ..models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF  
from ..model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from ..data.dataset import KITTIDataset
from ..training.renderer import volume_render
# ✅ rays.py 사용으로 변경
from ..utils.rays import get_rays, sample_points_on_rays

def calculate_iou_3d(box1, box2):
    """
    Calculates the 3D Intersection over Union (IoU) between two 3D bounding boxes.
    Each box is represented as [x, y, z, l, w, h, yaw].
    
    Note: This is a simplified IoU calculation. A more accurate version would
    handle rotations properly (e.g., using oriented bounding box intersection).
    """
    # Get the coordinates of the intersection rectangle
    x1_inter = max(box1[0] - box1[3]/2, box2[0] - box2[3]/2)
    y1_inter = max(box1[1] - box1[4]/2, box2[1] - box2[4]/2)
    z1_inter = max(box1[2] - box1[5]/2, box2[2] - box2[5]/2)
    
    x2_inter = min(box1[0] + box1[3]/2, box2[0] + box2[3]/2)
    y2_inter = min(box1[1] + box1[4]/2, box2[1] + box2[4]/2)
    z2_inter = min(box1[2] + box1[5]/2, box2[2] + box2[5]/2)
    
    # Calculate the volume of the intersection
    inter_l = max(0, x2_inter - x1_inter)
    inter_w = max(0, y2_inter - y1_inter)
    inter_h = max(0, z2_inter - z1_inter)
    intersection_volume = inter_l * inter_w * inter_h
    
    # Calculate the volume of each box
    box1_volume = box1[3] * box1[4] * box1[5]
    box2_volume = box2[3] * box2[4] * box2[5]
    
    # Calculate the volume of the union
    union_volume = box1_volume + box2_volume - intersection_volume
    
    # Calculate IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0
    return iou

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11):
    """Simplified SSIM calculation."""
    # This is a simplified version - in production, use skimage.metrics.structural_similarity
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return ssim

class Evaluator:
    """
    Handles the evaluation of 3D object detection models.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Resets the evaluation metrics."""
        self.total_iou = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_gts = []

    def add_batch(self, predictions, ground_truth):
        """
        Adds a batch of predictions and ground truth for evaluation.
        
        Args:
            predictions (list of dicts): Each dict contains 'boxes_3d' and 'scores'.
            ground_truth (list of dicts): Each dict contains 'boxes_3d'.
        """
        for pred, gt in zip(predictions, ground_truth):
            # For simplicity, we match the first predicted box to the first ground truth box
            if len(pred['boxes_3d']) > 0 and len(gt['boxes_3d']) > 0:
                iou = calculate_iou_3d(pred['boxes_3d'][0], gt['boxes_3d'][0])
                self.total_iou += iou
                self.num_samples += 1
            
            self.all_preds.append(pred)
            self.all_gts.append(gt)

    def evaluate(self):
        """
        Computes the final evaluation metrics.
        
        Returns:
            dict: A dictionary of evaluation metrics, e.g., mean IoU.
        """
        if self.num_samples == 0:
            return {'mean_iou_3d': 0}
            
        mean_iou = self.total_iou / self.num_samples
        
        # Here you could add more complex metrics like Average Precision (AP)
        # which would require matching predictions to ground truth more carefully.
        
        return {'mean_iou_3d': mean_iou}

class SSDNeRFBenchmark:
    """
    Comprehensive benchmark system for SSD-NeRF models.
    
    Evaluates both Dynamic and Static variants on multiple metrics:
    - 3D Object Detection Performance
    - NeRF Rendering Quality  
    - Temporal Consistency (for dynamic models)
    - Computational Performance
    """
    
    def __init__(self, config: dict, dataset: KITTIDataset):
        self.config = config
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation metrics storage
        self.results = {
            'dynamic_model': {},
            'static_model': {},
            'comparison': {}
        }
        
    def evaluate_model(self, 
                      model_path: str, 
                      model_type: str = 'dynamic',
                      max_samples: int = 100) -> Dict:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model_path: Path to model checkpoint
            model_type: 'dynamic' or 'static'
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Detailed evaluation results
        """
        print(f"🔍 Starting comprehensive evaluation of {model_type} SSD-NeRF")
        print(f"📂 Model: {model_path}")
        print(f"📊 Dataset: {len(self.dataset)} samples (evaluating {max_samples})")
        
        # Load model
        model = self._load_model(model_path, model_type)
        
        # Initialize metrics
        results = {
            'model_type': model_type,
            'model_path': model_path,
            'detection_metrics': {},
            'rendering_metrics': {},
            'performance_metrics': {},
            'temporal_metrics': {} if model_type == 'dynamic' else None
        }
        
        # Create dataloader
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        # Evaluation storage
        detection_results = []
        rendering_results = []
        performance_times = []
        temporal_results = [] if model_type == 'dynamic' else None
        
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_type}")):
                if idx >= max_samples:
                    break
                    
                start_time = time.time()
                
                if model_type == 'dynamic':
                    batch_results = self._evaluate_dynamic_batch(model, batch)
                else:
                    batch_results = self._evaluate_static_batch(model, batch)
                
                inference_time = time.time() - start_time
                performance_times.append(inference_time)
                
                # Store results
                if batch_results['detection_result']:
                    detection_results.append(batch_results['detection_result'])
                if batch_results['rendering_result']:
                    rendering_results.append(batch_results['rendering_result'])
                if temporal_results is not None and batch_results['temporal_result']:
                    temporal_results.append(batch_results['temporal_result'])
        
        # Compute aggregate metrics
        results['detection_metrics'] = self._compute_detection_metrics(detection_results)
        results['rendering_metrics'] = self._compute_rendering_metrics(rendering_results)
        results['performance_metrics'] = self._compute_performance_metrics(performance_times)
        
        if temporal_results:
            results['temporal_metrics'] = self._compute_temporal_metrics(temporal_results)
        
        # Store results
        self.results[f'{model_type}_model'] = results
        
        print(f"✅ {model_type.capitalize()} model evaluation complete")
        self._print_summary(results)
        
        return results
    
    def _load_model(self, model_path: str, model_type: str):
        """Load the appropriate model."""
        if model_type == 'dynamic':
            model = DynamicSSDNeRF(self.config).to(self.device)
        else:
            model = StaticSSDNeRF(num_classes=self.config['model']['ssd_nerf']['num_classes']).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _evaluate_dynamic_batch(self, model, batch) -> Dict:
        """Evaluate a single batch with dynamic model."""
        # Extract batch data
        images = batch['image'].to(self.device)
        lidar_points = batch['lidar_points'].to(self.device)
        scene_timestep = batch['scene_timestep'].to(self.device)
        
        B, C, H, W = images.shape
        
        # ✅ 개선된 Camera parameters 처리
        # dataset에서 focal과 c2w 가져오기
        if 'focal' in batch:
            focal = batch['focal'][0].item()
        else:
            focal = 721.5377  # KITTI default fallback
        
        if 'camera_to_world' in batch:
            c2w = batch['camera_to_world'][0]
            # c2w 차원 처리
            if c2w.shape == (4, 4):
                c2w_processed = c2w[:3, :4]  # 4x4 -> 3x4
            else:
                c2w_processed = c2w  # 이미 3x4
        else:
            # Fallback to identity
            c2w_processed = torch.eye(3, 4, dtype=torch.float32)
        
        c2w_processed = c2w_processed.to(self.device)
        
        # Generate rays (subsampled for evaluation speed)
        rays_o, rays_d = get_rays(H, W, focal, c2w_processed)
        
        # Subsample for speed
        num_rays = min(2048, rays_o.shape[0])
        ray_indices = torch.randperm(rays_o.shape[0])[:num_rays]
        rays_o_sub = rays_o[ray_indices]
        rays_d_sub = rays_d[ray_indices]
        
        # ✅ rays.py의 sample_points_on_rays 사용 (named parameters)
        pts, z_vals = sample_points_on_rays(
            rays_o=rays_o_sub, 
            rays_d=rays_d_sub, 
            near=0.5, 
            far=50.0, 
            n_samples=64,
            perturb=False,  # evaluation에서는 deterministic
            l_disp=True     # disparity space sampling
        )
        pts_batch = pts.unsqueeze(0)
        view_dirs_batch = rays_d_sub.unsqueeze(0)
        
        # Model forward pass
        diffusion_timesteps = torch.randint(0, self.config['model']['diffusion']['time_steps'], (B,)).to(self.device)
        outputs = model(lidar_points, view_dirs_batch, pts_batch, diffusion_timesteps, scene_timestep)
        
        # Render
        raw_output = outputs['nerf_output'].squeeze(0)
        rgb_map, depth_map, acc_map, disp_map = volume_render(raw_output, z_vals, rays_d_sub)
        
        # Extract results
        detection_result = self._extract_dynamic_detections(outputs, pts_batch)
        rendering_result = self._extract_rendering_quality(rgb_map, images, ray_indices, H, W)
        temporal_result = self._extract_temporal_consistency(outputs, scene_timestep)
        
        return {
            'detection_result': detection_result,
            'rendering_result': rendering_result,
            'temporal_result': temporal_result
        }
    
    def _evaluate_static_batch(self, model, batch) -> Dict:
        """Evaluate a single batch with static model."""
        images = batch['image'].to(self.device)
        
        # Forward pass
        locs_2d, confs_2d, pred_3d_params = model(images, None)
        
        # Extract results
        detection_result = self._extract_static_detections(locs_2d, confs_2d, pred_3d_params)
        rendering_result = None  # Static model doesn't render
        temporal_result = None   # Static model doesn't handle temporal data
        
        return {
            'detection_result': detection_result,
            'rendering_result': rendering_result,
            'temporal_result': temporal_result
        }
    
    def _extract_dynamic_detections(self, outputs: Dict, pts: torch.Tensor) -> Dict:
        """Extract detection metrics from dynamic model outputs."""
        displacement = outputs['displacement']
        displacement_magnitude = torch.norm(displacement, dim=-1)
        
        # Count dynamic objects
        dynamic_threshold = 0.1
        dynamic_mask = displacement_magnitude > dynamic_threshold
        num_dynamic_objects = (dynamic_mask.sum() > 50).sum().item()  # Clusters with >50 points
        
        return {
            'num_detections': num_dynamic_objects,
            'max_displacement': torch.max(displacement_magnitude).item(),
            'mean_displacement': torch.mean(displacement_magnitude).item(),
            'dynamic_ratio': dynamic_mask.float().mean().item()
        }
    
    def _extract_static_detections(self, locs_2d, confs_2d, pred_3d_params) -> Dict:
        """Extract detection metrics from static model outputs."""
        # Simple detection counting
        scores = torch.softmax(confs_2d, dim=-1)
        best_scores, _ = scores.max(dim=-1)
        valid_detections = (best_scores > 0.5).sum().item()
        
        return {
            'num_detections': valid_detections,
            'max_confidence': torch.max(best_scores).item(),
            'mean_confidence': torch.mean(best_scores).item(),
            'detection_diversity': torch.std(best_scores).item()
        }
    
    def _extract_rendering_quality(self, rendered_rgb, original_images, ray_indices, H, W) -> Dict:
        """Extract rendering quality metrics."""
        # Reconstruct full image for comparison
        full_rendered = torch.zeros(H * W, 3, device=rendered_rgb.device)
        full_rendered[ray_indices] = rendered_rgb
        full_rendered = full_rendered.reshape(H, W, 3)
        
        # Convert to numpy for metric calculation
        rendered_np = full_rendered.cpu().numpy()
        original_np = original_images.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Calculate metrics
        psnr = calculate_psnr(rendered_np, original_np)
        ssim = calculate_ssim(rendered_np.mean(axis=2), original_np.mean(axis=2))
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'mse': np.mean((rendered_np - original_np) ** 2),
            'rendering_coverage': len(ray_indices) / (H * W)
        }
    
    def _extract_temporal_consistency(self, outputs: Dict, scene_timestep: torch.Tensor) -> Dict:
        """Extract temporal consistency metrics."""
        displacement = outputs['displacement']
        features = outputs['diffusion_features']
        
        return {
            'temporal_variance': torch.var(displacement).item(),
            'feature_stability': torch.norm(features).item(),
            'timestep': scene_timestep.item()
        }
    
    def _compute_detection_metrics(self, detection_results: List[Dict]) -> Dict:
        """Compute aggregate detection metrics."""
        if not detection_results:
            return {}
        
        num_detections = [r['num_detections'] for r in detection_results]
        
        metrics = {
            'avg_detections_per_frame': np.mean(num_detections),
            'max_detections_per_frame': np.max(num_detections),
            'total_detections': np.sum(num_detections),
            'detection_rate': np.mean([d > 0 for d in num_detections])
        }
        
        # Add model-specific metrics
        if 'max_displacement' in detection_results[0]:
            # Dynamic model metrics
            metrics.update({
                'avg_max_displacement': np.mean([r['max_displacement'] for r in detection_results]),
                'avg_dynamic_ratio': np.mean([r['dynamic_ratio'] for r in detection_results])
            })
        
        if 'max_confidence' in detection_results[0]:
            # Static model metrics
            metrics.update({
                'avg_max_confidence': np.mean([r['max_confidence'] for r in detection_results]),
                'avg_confidence': np.mean([r['mean_confidence'] for r in detection_results])
            })
        
        return metrics
    
    def _compute_rendering_metrics(self, rendering_results: List[Dict]) -> Dict:
        """Compute aggregate rendering metrics."""
        if not rendering_results:
            return {}
        
        return {
            'avg_psnr': np.mean([r['psnr'] for r in rendering_results if np.isfinite(r['psnr'])]),
            'avg_ssim': np.mean([r['ssim'] for r in rendering_results]),
            'avg_mse': np.mean([r['mse'] for r in rendering_results]),
            'avg_coverage': np.mean([r['rendering_coverage'] for r in rendering_results])
        }
    
    def _compute_performance_metrics(self, performance_times: List[float]) -> Dict:
        """Compute performance metrics."""
        times = np.array(performance_times)
        
        return {
            'avg_inference_time': float(np.mean(times)),
            'max_inference_time': float(np.max(times)),
            'min_inference_time': float(np.min(times)),
            'std_inference_time': float(np.std(times)),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_samples': len(times)
        }
    
    def _compute_temporal_metrics(self, temporal_results: List[Dict]) -> Dict:
        """Compute temporal consistency metrics."""
        if not temporal_results:
            return {}
        
        return {
            'avg_temporal_variance': np.mean([r['temporal_variance'] for r in temporal_results]),
            'avg_feature_stability': np.mean([r['feature_stability'] for r in temporal_results]),
            'temporal_range': [min(r['timestep'] for r in temporal_results),
                              max(r['timestep'] for r in temporal_results)]
        }
    
    def compare_models(self, 
                      dynamic_checkpoint: str, 
                      static_checkpoint: str,
                      max_samples: int = 50) -> Dict:
        """
        Compare Dynamic vs Static SSD-NeRF models side by side.
        
        Args:
            dynamic_checkpoint: Path to dynamic model checkpoint
            static_checkpoint: Path to static model checkpoint
            max_samples: Number of samples for comparison
            
        Returns:
            Comprehensive comparison results
        """
        print("🥊 Starting head-to-head model comparison")
        
        # Evaluate both models
        dynamic_results = self.evaluate_model(dynamic_checkpoint, 'dynamic', max_samples)
        static_results = self.evaluate_model(static_checkpoint, 'static', max_samples)
        
        # Compute comparison metrics
        comparison = {
            'performance_comparison': self._compare_performance(dynamic_results, static_results),
            'detection_comparison': self._compare_detection(dynamic_results, static_results),
            'capability_comparison': self._compare_capabilities(dynamic_results, static_results),
            'recommendation': self._generate_recommendation(dynamic_results, static_results)
        }
        
        self.results['comparison'] = comparison
        
        print("📊 Model comparison complete")
        self._print_comparison_summary(comparison)
        
        return comparison
    
    def _compare_performance(self, dynamic_results: Dict, static_results: Dict) -> Dict:
        """Compare performance metrics."""
        dynamic_perf = dynamic_results['performance_metrics']
        static_perf = static_results['performance_metrics']
        
        return {
            'speed_ratio': static_perf['avg_fps'] / dynamic_perf['avg_fps'],
            'dynamic_fps': dynamic_perf['avg_fps'],
            'static_fps': static_perf['avg_fps'],
            'faster_model': 'static' if static_perf['avg_fps'] > dynamic_perf['avg_fps'] else 'dynamic'
        }
    
    def _compare_detection(self, dynamic_results: Dict, static_results: Dict) -> Dict:
        """Compare detection capabilities."""
        dynamic_det = dynamic_results['detection_metrics']
        static_det = static_results['detection_metrics']
        
        return {
            'dynamic_detections': dynamic_det.get('avg_detections_per_frame', 0),
            'static_detections': static_det.get('avg_detections_per_frame', 0),
            'detection_advantage': 'dynamic' if dynamic_det.get('avg_detections_per_frame', 0) > static_det.get('avg_detections_per_frame', 0) else 'static'
        }
    
    def _compare_capabilities(self, dynamic_results: Dict, static_results: Dict) -> Dict:
        """Compare model capabilities."""
        capabilities = {
            'dynamic_advantages': [
                '3D Scene Reconstruction',
                'Temporal Modeling', 
                'Dense Environment Mapping',
                'Dynamic Object Tracking'
            ],
            'static_advantages': [
                'Faster Inference',
                'Lower Memory Usage',
                'Simpler Deployment',
                'Traditional CV Pipeline Compatible'
            ],
            'rendering_quality': 'dynamic_only' if dynamic_results['rendering_metrics'] else 'none'
        }
        
        return capabilities
    
    def _generate_recommendation(self, dynamic_results: Dict, static_results: Dict) -> Dict:
        """Generate usage recommendations."""
        dynamic_fps = dynamic_results['performance_metrics']['avg_fps']
        static_fps = static_results['performance_metrics']['avg_fps']
        
        if dynamic_fps > 10:  # Real-time capable
            recommendation = "dynamic"
            reason = "Dynamic model achieves real-time performance with superior 3D understanding"
        elif static_fps > 20:  # High-speed requirement
            recommendation = "static"  
            reason = "Static model better for high-speed requirements where basic detection suffices"
        else:
            recommendation = "hybrid"
            reason = "Consider hybrid approach: static for detection, dynamic for detailed analysis"
        
        return {
            'recommended_model': recommendation,
            'reasoning': reason,
            'use_cases': {
                'autonomous_driving': 'dynamic' if dynamic_fps > 10 else 'static',
                'robotics': 'dynamic',
                'surveillance': 'static',
                'ar_vr': 'dynamic'
            }
        }
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print(f"\n📊 {results['model_type'].upper()} MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        # Performance
        perf = results['performance_metrics']
        print(f"⚡ Performance: {perf['avg_fps']:.1f} FPS (avg: {perf['avg_inference_time']:.3f}s)")
        
        # Detection
        det = results['detection_metrics']
        print(f"🎯 Detection: {det.get('avg_detections_per_frame', 0):.1f} objects/frame")
        
        # Rendering (if available)
        if results['rendering_metrics']:
            rend = results['rendering_metrics']
            print(f"🖼️  Rendering: PSNR={rend['avg_psnr']:.2f}, SSIM={rend['avg_ssim']:.3f}")
        
        # Temporal (if available)
        if results['temporal_metrics']:
            temp = results['temporal_metrics']
            print(f"⏱️  Temporal: Variance={temp['avg_temporal_variance']:.4f}")
    
    def _print_comparison_summary(self, comparison: Dict):
        """Print comparison summary."""
        print(f"\n🥊 MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        perf_comp = comparison['performance_comparison']
        print(f"🏃 Speed: {perf_comp['faster_model'].upper()} wins ({perf_comp['speed_ratio']:.1f}x faster)")
        
        det_comp = comparison['detection_comparison']  
        print(f"🎯 Detection: {det_comp['detection_advantage'].upper()} advantage")
        
        rec = comparison['recommendation']
        print(f"💡 Recommendation: {rec['recommended_model'].upper()}")
        print(f"   Reason: {rec['reasoning']}")
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")

# Legacy function for backward compatibility
def evaluate_model(model, dataset, config, model_type='dynamic'):
    """
    Legacy function - use SSDNeRFBenchmark for comprehensive evaluation.
    """
    benchmark = SSDNeRFBenchmark(config, dataset)
    # This would need a model path, so we'll create a dummy one
    return benchmark.evaluate_model("dummy_path.pth", model_type, max_samples=10)
