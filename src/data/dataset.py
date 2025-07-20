import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from PIL import Image
import glob
from torchvision import transforms

class KITTIDataset(Dataset):
    """
    KITTI Dataset Loader
    - Loads images, LiDAR point clouds, and calibration data.
    - Generates a normalized time step for each frame to enable temporal modeling.
    """
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.data_path = config['data']['path']
        
        self.image_dir = os.path.join(self.data_path, split, 'image_2')
        self.lidar_dir = os.path.join(self.data_path, split, 'velodyne')
        self.calib_dir = os.path.join(self.data_path, split, 'calib')
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # Calculate sequence length for time normalization
        self.sequence_length = len(self.image_files)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        
        # LiDAR data
        lidar_path = os.path.join(self.lidar_dir, self.image_files[idx].replace('.png', '.bin'))
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # Calibration data
        calib_path = os.path.join(self.calib_dir, self.image_files[idx].replace('.png', '.txt'))
        calib = self._load_calib(calib_path)
        focal = calib['P2'][0, 0]
        # This is a simplified c2w, real applications might need full SE(3) transform
        c2w = np.eye(4) # Placeholder

        # --- Key Enhancement: Add normalized time step ---
        # Extract frame index from filename (e.g., '000001.png' -> 1)
        frame_idx = int(os.path.splitext(self.image_files[idx])[0])
        # Normalize by sequence length to get t in [0, 1]
        normalized_time = frame_idx / (self.sequence_length - 1) if self.sequence_length > 1 else 0.0

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'lidar_points': torch.from_numpy(lidar_points[:, :3]), # Use x, y, z
            'focal': torch.tensor(focal, dtype=torch.float32),
            'camera_to_world': torch.from_numpy(c2w).float(),  # 키 이름 수정
            'scene_timestep': torch.tensor([normalized_time], dtype=torch.float32),
            'sample_id': self.image_files[idx]  # 추가로 sample_id도 추가
        }
        
        return sample

    def _load_calib(self, filepath):
        """Loads calibration data from a KITTI calib file."""
        calib = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.strip().split()]).reshape(3, 4) if key in ['P0', 'P1', 'P2', 'P3'] else np.array([float(x) for x in value.strip().split()])
        # Simplified example for camera-to-velo transform
        # In a real scenario, you'd parse the full rigid body transformation
        calib['T_cam_velo'] = np.eye(4)
        return calib

def download_kitti_sample():
    """
    KITTI 샘플 데이터를 다운로드하는 함수
    """
    print("KITTI 데이터셋 다운로드 안내:")
    print("1. KITTI 공식 웹사이트 방문: http://www.cvlibs.net/datasets/kitti/")
    print("2. '3D Object Detection' 섹션에서 다음 파일들 다운로드:")
    print("   - Left color images of object data set (12 GB)")
    print("   - Velodyne point clouds (29 GB)")
    print("   - Camera calibration matrices of object data set (16 MB)")
    print("3. 다운로드한 파일들을 다음 구조로 압축 해제:")
    print("   data/kitti/object/")
    print("   ├── training/")
    print("   │   ├── image_2/")
    print("   │   ├── velodyne/")
    print("   │   └── calib/")
    print("   └── testing/")
    print("       ├── image_2/")
    print("       ├── velodyne/")
    print("       └── calib/")

if __name__ == '__main__':
    from configs.default_config import config
    
    # KITTI 데이터 경로 확인
    if os.path.exists(config['data']['path']):
        try:
            dataset = KITTIDataset(config)
            print(f"KITTI 데이터셋 로드 성공: {len(dataset)}개 샘플")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print("\n--- 샘플 데이터 ---")
                print(f"이미지 크기: {sample['image'].shape}")
                print(f"LiDAR 포인트 크기: {sample['lidar_points'].shape}")
                print(f"샘플 ID: {sample['sample_id']}")
                print(f"Scene timestep: {sample['scene_timestep']}")
            else:
                print("데이터셋이 비어있습니다.")
                download_kitti_sample()
        except Exception as e:
            print(f"데이터셋 로드 중 오류: {e}")
            download_kitti_sample()
    else:
        print(f"KITTI 데이터 경로를 찾을 수 없습니다: {config['data']['path']}")
        download_kitti_sample() 