"""
SSD-NeRF Autonomous Vehicle Integration Example
자율주행차에 SSD-NeRF 모델을 통합하는 방법을 보여주는 예시
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import our SSD-NeRF models
from src.models.ssd_nerf import SSD_NeRF as DynamicSSDNeRF
from src.model_arch.ssd_nerf_model import SSDNeRF as StaticSSDNeRF
from src.training.renderer import volume_render, sample_along_rays
from src.utils.ray_utils import get_rays

@dataclass
class SensorData:
    """자율주행차 센서 데이터 구조"""
    camera_rgb: np.ndarray      # (H, W, 3) RGB 카메라
    lidar_points: np.ndarray    # (N, 3) LiDAR 포인트 클라우드
    timestamp: float            # 타임스탬프
    vehicle_pose: np.ndarray    # (4, 4) 차량 위치/자세
    velocity: float             # 차량 속도

@dataclass  
class DetectionResult:
    """3D 객체 검출 결과"""
    boxes_3d: List[np.ndarray]  # 3D 바운딩 박스들
    classes: List[str]          # 객체 클래스 (Car, Pedestrian, etc.)
    scores: List[float]         # 신뢰도 점수
    track_ids: List[int]        # 객체 추적 ID

class AutonomousVehiclePerception:
    """
    자율주행차용 SSD-NeRF 인식 시스템
    
    실시간으로 센서 데이터를 처리하여:
    1. 3D 환경 재구성
    2. 동적 객체 검출 및 추적
    3. 안전 위험 평가
    4. 경로 계획 지원
    """
    
    def __init__(self, config: dict, model_path: str, use_dynamic: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.use_dynamic = use_dynamic
        
        # 모델 초기화
        if use_dynamic:
            print("🚗 Initializing Dynamic SSD-NeRF for autonomous driving")
            self.model = DynamicSSDNeRF(config).to(self.device)
        else:
            print("🚗 Initializing Static SSD-NeRF for autonomous driving")
            self.model = StaticSSDNeRF(num_classes=config['model']['ssd_nerf']['num_classes']).to(self.device)
            
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 추적 시스템 초기화
        self.object_tracker = {}
        self.next_track_id = 0
        
        # 성능 모니터링
        self.inference_times = []
        
    def process_sensor_data(self, sensor_data: SensorData) -> Dict:
        """
        실시간 센서 데이터 처리
        
        Args:
            sensor_data: 차량의 센서 데이터
            
        Returns:
            Dict: 처리 결과 (3D 검출, 환경 맵, 위험 평가)
        """
        start_time = time.time()
        
        with torch.no_grad():
            if self.use_dynamic:
                result = self._process_dynamic_model(sensor_data)
            else:
                result = self._process_static_model(sensor_data)
                
        # 성능 모니터링
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 실시간 요구사항 체크 (자율주행은 보통 10Hz 이상 필요)
        if inference_time > 0.1:  # 100ms 초과시 경고
            print(f"⚠️  Inference time too slow: {inference_time:.3f}s")
            
        result['inference_time'] = inference_time
        result['fps'] = 1.0 / inference_time
        
        return result
    
    def _process_dynamic_model(self, sensor_data: SensorData) -> Dict:
        """동적 SSD-NeRF 모델로 처리"""
        
        # 입력 데이터 준비
        rgb_tensor = torch.from_numpy(sensor_data.camera_rgb).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        
        lidar_tensor = torch.from_numpy(sensor_data.lidar_points).float()
        lidar_tensor = lidar_tensor.unsqueeze(0).to(self.device)
        
        # 시간 정보 (동적 객체 모델링을 위함)
        scene_timestep = torch.tensor([sensor_data.timestamp % 1.0]).unsqueeze(0).to(self.device)
        
        # 카메라 파라미터
        H, W = rgb_tensor.shape[2], rgb_tensor.shape[3]
        focal = 721.5  # KITTI 기본값 (실제로는 캘리브레이션에서 가져와야 함)
        c2w = torch.from_numpy(sensor_data.vehicle_pose).float().to(self.device)
        
        # Ray 생성
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o, rays_d = rays_o.to(self.device), rays_d.to(self.device)
        
        # 실시간 처리를 위한 ray 서브샘플링
        num_rays = min(2048, rays_o.shape[0])  # 성능을 위해 제한
        ray_indices = torch.randperm(rays_o.shape[0])[:num_rays]
        rays_o_sample = rays_o[ray_indices]
        rays_d_sample = rays_d[ray_indices]
        
        # 3D 포인트 샘플링
        pts, z_vals = sample_along_rays(rays_o_sample, rays_d_sample, 0.5, 50.0, 64)
        pts = pts.unsqueeze(0)
        view_dirs = rays_d_sample.unsqueeze(0)
        
        # 디퓨전 타임스텝
        diffusion_timesteps = torch.randint(0, self.config['model']['diffusion']['time_steps'], (1,)).to(self.device)
        
        # 모델 추론
        outputs = self.model(lidar_tensor, view_dirs, pts, diffusion_timesteps, scene_timestep)
        
        # 3D 환경 렌더링
        raw_output = outputs['nerf_output'].squeeze(0)
        rgb_map, depth_map, acc_map, disp_map = volume_render(raw_output, z_vals, rays_d_sample)
        
        # 동적 객체 검출 (displacement 분석)
        displacement = outputs['displacement']
        moving_objects = self._detect_moving_objects(displacement, pts, threshold=0.1)
        
        return {
            'detections': moving_objects,
            'rendered_view': rgb_map.cpu().numpy(),
            'depth_map': depth_map.cpu().numpy(),
            'environment_features': outputs['diffusion_features'].cpu().numpy(),
            'dynamic_displacement': displacement.cpu().numpy()
        }
    
    def _process_static_model(self, sensor_data: SensorData) -> Dict:
        """정적 SSD-NeRF 모델로 처리"""
        
        rgb_tensor = torch.from_numpy(sensor_data.camera_rgb).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        
        # 2D 검출 수행
        locs_2d, confs_2d, pred_3d_params = self.model(rgb_tensor, None)
        
        # 검출 결과 후처리
        detections = self._postprocess_static_detections(locs_2d, confs_2d, pred_3d_params)
        
        return {
            'detections': detections,
            'rendered_view': sensor_data.camera_rgb,  # 원본 이미지
            'depth_map': None,
            'environment_features': None,
            'dynamic_displacement': None
        }
    
    def _detect_moving_objects(self, displacement: torch.Tensor, pts: torch.Tensor, threshold: float = 0.1) -> DetectionResult:
        """변위 분석을 통한 동적 객체 검출"""
        
        # 변위 크기 계산
        displacement_magnitude = torch.norm(displacement, dim=-1)
        
        # 움직이는 포인트 마스크
        moving_mask = displacement_magnitude > threshold
        
        # 클러스터링을 통한 객체 분리 (간단한 구현)
        moving_points = pts.squeeze(0)[moving_mask.squeeze(0)]
        
        if moving_points.shape[0] > 0:
            # 간단한 바운딩 박스 생성
            min_coords = torch.min(moving_points, dim=0)[0]
            max_coords = torch.max(moving_points, dim=0)[0]
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            
            # 3D 바운딩 박스 [x, y, z, l, w, h, yaw]
            bbox_3d = torch.cat([center, size, torch.tensor([0.0], device=self.device)])
            
            return DetectionResult(
                boxes_3d=[bbox_3d.cpu().numpy()],
                classes=['Moving_Object'],
                scores=[0.8],
                track_ids=[self._get_track_id()]
            )
        else:
            return DetectionResult([], [], [], [])
    
    def _postprocess_static_detections(self, locs_2d, confs_2d, pred_3d_params) -> DetectionResult:
        """정적 모델 검출 결과 후처리"""
        
        # 간단한 후처리 (실제로는 더 복잡한 NMS 등이 필요)
        if pred_3d_params is not None:
            boxes_3d = pred_3d_params.cpu().numpy()
            scores = torch.softmax(confs_2d, dim=-1).max(dim=-1)[0].cpu().numpy()
            classes = ['Vehicle'] * len(boxes_3d)
            track_ids = [self._get_track_id() for _ in range(len(boxes_3d))]
            
            return DetectionResult(
                boxes_3d=boxes_3d.tolist(),
                classes=classes,
                scores=scores.tolist(),
                track_ids=track_ids
            )
        else:
            return DetectionResult([], [], [], [])
    
    def _get_track_id(self) -> int:
        """새로운 추적 ID 생성"""
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
    
    def get_performance_stats(self) -> Dict:
        """성능 통계 반환"""
        if not self.inference_times:
            return {}
            
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'avg_fps': 1.0 / np.mean(self.inference_times),
            'total_frames': len(self.inference_times)
        }

# 실제 자율주행 시스템 통합 예시
def autonomous_driving_integration_example():
    """
    자율주행 시스템 통합 예시
    """
    
    # 설정 로드
    from src.utils.config_utils import load_config
    config = load_config('configs/default_config.py')
    
    # 인식 시스템 초기화
    perception_system = AutonomousVehiclePerception(
        config=config,
        model_path='output/checkpoints/model_epoch_100.pth',
        use_dynamic=True
    )
    
    print("🚗 자율주행 인식 시스템 초기화 완료")
    print("📡 센서 데이터 처리 시작...")
    
    # 시뮬레이션된 센서 데이터
    for frame in range(10):
        # 가상의 센서 데이터 생성
        sensor_data = SensorData(
            camera_rgb=np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8),
            lidar_points=np.random.randn(8192, 3).astype(np.float32),
            timestamp=frame * 0.1,  # 10Hz
            vehicle_pose=np.eye(4, dtype=np.float32),
            velocity=50.0  # km/h
        )
        
        # 실시간 처리
        result = perception_system.process_sensor_data(sensor_data)
        
        print(f"Frame {frame}: {result['fps']:.1f} FPS, "
              f"{len(result['detections'].boxes_3d)} objects detected")
    
    # 성능 통계
    stats = perception_system.get_performance_stats()
    print("\n📊 성능 통계:")
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}")

if __name__ == "__main__":
    autonomous_driving_integration_example() 