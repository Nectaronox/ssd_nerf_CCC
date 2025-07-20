#!/usr/bin/env python3
"""
KITTI 더미 데이터 생성 스크립트
실제 KITTI 데이터가 없을 때 테스트용 더미 데이터를 생성합니다.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_utils import load_config

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_dummy_kitti_data(data_path, splits=['train', 'test'], num_samples=10):
    """
    KITTI 구조의 더미 데이터 생성
    
    Args:
        data_path (str): 데이터 생성 경로
        splits (list): 생성할 split 리스트 ('train', 'test')
        num_samples (int): 각 split별 샘플 수
    """
    logger = setup_logging()
    
    logger.info(f"🚀 KITTI 더미 데이터 생성 시작")
    logger.info(f"📁 경로: {data_path}")
    logger.info(f"📊 Split: {splits}")
    logger.info(f"🔢 샘플 수: {num_samples}")
    
    # KITTI calibration 템플릿
    calib_template = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""
    
    for split in splits:
        logger.info(f"📂 {split} split 생성 중...")
        
        # 디렉토리 생성
        image_dir = os.path.join(data_path, split, 'image_2')
        lidar_dir = os.path.join(data_path, split, 'velodyne')
        calib_dir = os.path.join(data_path, split, 'calib')
        
        for dir_path in [image_dir, lidar_dir, calib_dir]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"  📁 생성: {dir_path}")
        
        # 샘플 데이터 생성
        for i in range(num_samples):
            # 더미 이미지 생성 (KITTI 표준 크기)
            # 다양한 패턴으로 생성하여 구별 가능하게 함
            pattern = i % 3
            if pattern == 0:
                # 체크보드 패턴
                dummy_image = np.zeros((375, 1242, 3), dtype=np.uint8)
                dummy_image[::20, ::20] = [255, 255, 255]
                dummy_image += np.random.randint(0, 50, dummy_image.shape, dtype=np.uint8)
            elif pattern == 1:
                # 그라데이션 패턴
                dummy_image = np.zeros((375, 1242, 3), dtype=np.uint8)
                for j in range(375):
                    dummy_image[j, :, :] = int(255 * j / 375)
                dummy_image += np.random.randint(0, 30, dummy_image.shape, dtype=np.uint8)
            else:
                # 노이즈 패턴
                dummy_image = np.random.randint(50, 200, (375, 1242, 3), dtype=np.uint8)
            
            image_path = os.path.join(image_dir, f"{i:06d}.png")
            Image.fromarray(dummy_image).save(image_path)
            
            # 더미 LiDAR 데이터 생성 (실제적인 3D 분포)
            # 차량 주변의 리얼리스틱한 점군 분포
            num_points = np.random.randint(800, 1200)
            
            # 전방 도로와 주변 물체를 시뮬레이션
            x = np.random.normal(0, 20, num_points)  # 좌우 분포
            y = np.random.uniform(-2, 1, num_points)  # 높이 (도로면 ~ 차량 높이)
            z = np.random.uniform(2, 50, num_points)  # 전방 거리
            intensity = np.random.uniform(0, 1, num_points)  # 반사 강도
            
            dummy_lidar = np.column_stack([x, y, z, intensity]).astype(np.float32)
            lidar_path = os.path.join(lidar_dir, f"{i:06d}.bin")
            dummy_lidar.tofile(lidar_path)
            
            # Calibration 파일 생성
            calib_path = os.path.join(calib_dir, f"{i:06d}.txt")
            with open(calib_path, 'w') as f:
                f.write(calib_template)
        
        logger.info(f"✅ {split} split 완료: {num_samples}개 샘플")
    
    logger.info("🎉 더미 데이터 생성 완료!")
    logger.info(f"📁 생성된 경로: {data_path}")
    logger.info("💡 사용 방법:")
    logger.info("  python scripts/run_evaluation.py --config configs/default_config.py --use_dummy_data")
    logger.info("  python scripts/run_inference.py --config configs/default_config.py --checkpoint model.pth --use_dummy_data")

def main():
    parser = argparse.ArgumentParser(
        description="🎲 KITTI 더미 데이터 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 사용법:
  # 기본 더미 데이터 생성
  python scripts/create_dummy_data.py

  # 설정 파일 지정
  python scripts/create_dummy_data.py --config configs/default_config.py

  # 더 많은 샘플 생성
  python scripts/create_dummy_data.py --num_samples 50

  # 특정 경로에 생성
  python scripts/create_dummy_data.py --data_path /path/to/dummy_kitti
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/default_config.py',
                        help="설정 파일 경로")
    parser.add_argument('--data_path', type=str, default=None,
                        help="더미 데이터 생성 경로 (None이면 config에서 가져옴)")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="각 split별 생성할 샘플 수")
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        choices=['train', 'test'],
                        help="생성할 split 리스트")
    
    args = parser.parse_args()
    
    try:
        # 설정 로드
        if os.path.exists(args.config):
            config = load_config(args.config)
            data_path = args.data_path or config['data']['path']
        else:
            data_path = args.data_path or 'data/kitti'
            print(f"⚠️ 설정 파일을 찾을 수 없어 기본 경로 사용: {data_path}")
        
        # 더미 데이터 생성
        create_dummy_kitti_data(data_path, args.splits, args.num_samples)
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"❌ 더미 데이터 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 