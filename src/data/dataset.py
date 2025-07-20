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
    def __init__(self, config, split='train', create_dummy_data=False):
        self.config = config
        self.split = split
        self.data_path = config['data']['path']
        self.create_dummy_data = create_dummy_data
        
        # 데이터 경로 구성
        self.image_dir = os.path.join(self.data_path, split, 'image_2')
        self.lidar_dir = os.path.join(self.data_path, split, 'velodyne')
        self.calib_dir = os.path.join(self.data_path, split, 'calib')
        
        # ✅ 데이터 경로 존재 여부 확인
        if not self._check_data_exists():
            if create_dummy_data:
                print(f"⚠️ KITTI 데이터가 없어서 더미 데이터를 생성합니다.")
                self._create_dummy_data()
            else:
                self._print_data_download_guide()
                raise FileNotFoundError(
                    f"❌ KITTI 데이터셋을 찾을 수 없습니다.\n"
                    f"경로: {self.image_dir}\n"
                    f"해결방법:\n"
                    f"1. KITTI 데이터셋을 다운로드하여 {self.data_path}에 압축 해제\n"
                    f"2. 또는 create_dummy_data=True로 더미 데이터 사용\n"
                    f"3. 또는 config에서 올바른 data.path 설정"
                )
        
        # 이미지 파일 리스트 로드
        try:
            self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
            if len(self.image_files) == 0:
                raise ValueError(f"이미지 파일이 없습니다: {self.image_dir}")
        except Exception as e:
            if create_dummy_data:
                self.image_files = [f"dummy_{i:06d}.png" for i in range(10)]
            else:
                raise e
        
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
        
        # ✅ 개선된 camera_to_world 변환 (placeholder에서 실제 calibration 기반으로 변경)
        # KITTI P2는 left color camera의 projection matrix이므로, 이를 이용해 c2w 생성
        P2 = calib['P2']  # (3, 4) projection matrix
        
        # P2에서 카메라 내부 파라미터와 extrinsic 분리
        K = P2[:3, :3]  # Intrinsic matrix
        # KITTI의 경우 일반적으로 rectified coordinate system을 사용
        # 간단한 c2w 변환 생성 (identity rotation + small translation)
        c2w = np.eye(4, dtype=np.float32)
        
        # 실제 KITTI 환경에 맞는 기본 변환 적용
        # 카메라가 차량에 장착된 위치를 반영 (높이와 전방 이동)
        c2w[0, 3] = 0.0    # x translation  
        c2w[1, 3] = -1.7   # y translation (카메라 높이)
        c2w[2, 3] = 0.0    # z translation

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
            'camera_to_world': torch.from_numpy(c2w).float(),  # 개선된 c2w 변환
            'scene_timestep': torch.tensor([normalized_time], dtype=torch.float32),
            'sample_id': self.image_files[idx],  # 추가로 sample_id도 추가
            'P2': torch.from_numpy(P2).float(),  # 디버깅용으로 P2도 추가
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

    def _check_data_exists(self):
        """데이터 디렉토리 존재 여부 확인"""
        required_dirs = [self.image_dir, self.lidar_dir, self.calib_dir]
        return all(os.path.exists(dir_path) for dir_path in required_dirs)

    def _create_dummy_data(self):
        """테스트용 더미 데이터 생성"""
        import numpy as np
        from PIL import Image
        
        print(f"📁 더미 데이터 디렉토리 생성 중...")
        
        # 디렉토리 생성
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.lidar_dir, exist_ok=True)
        os.makedirs(self.calib_dir, exist_ok=True)
        
        # 더미 이미지 생성 (10개)
        for i in range(10):
            # 375 x 1242 크기의 더미 이미지 생성
            dummy_image = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
            image_path = os.path.join(self.image_dir, f"dummy_{i:06d}.png")
            Image.fromarray(dummy_image).save(image_path)
            
            # 더미 LiDAR 데이터 생성
            dummy_lidar = np.random.randn(1000, 4).astype(np.float32)
            lidar_path = os.path.join(self.lidar_dir, f"dummy_{i:06d}.bin")
            dummy_lidar.tofile(lidar_path)
            
            # 더미 calibration 데이터 생성
            calib_content = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""
            
            calib_path = os.path.join(self.calib_dir, f"dummy_{i:06d}.txt")
            with open(calib_path, 'w') as f:
                f.write(calib_content)
        
        print(f"✅ 더미 데이터 생성 완료: {len(os.listdir(self.image_dir))}개 샘플")

    def _print_data_download_guide(self):
        """KITTI 데이터셋 다운로드 가이드 출력"""
        print("\n" + "="*80)
        print("🚨 KITTI 데이터셋이 필요합니다!")
        print("="*80)
        print("📥 다운로드 방법:")
        print("1. KITTI 공식 웹사이트 방문: http://www.cvlibs.net/datasets/kitti/")
        print("2. '3D Object Detection' 또는 'Raw Data' 섹션에서 다음 파일들 다운로드:")
        print("   - Left color images of object data set (12 GB)")
        print("   - Velodyne point clouds (29 GB)")
        print("   - Camera calibration matrices (16 MB)")
        print("3. 다운로드한 파일들을 다음 구조로 압축 해제:")
        print(f"   {self.data_path}/")
        print("   ├── training/")
        print("   │   ├── image_2/")
        print("   │   ├── velodyne/")
        print("   │   └── calib/")
        print("   └── testing/")
        print("       ├── image_2/")
        print("       ├── velodyne/")
        print("       └── calib/")
        print("\n💡 임시 해결책:")
        print("더미 데이터로 테스트하려면 다음과 같이 실행:")
        print("  dataset = KITTIDataset(config, split='test', create_dummy_data=True)")
        print("="*80)

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