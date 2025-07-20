#!/usr/bin/env python3
"""
무료 KITTI 형식 샘플 데이터 다운로드 스크립트
공개 소스에서 테스트용 데이터를 가져옵니다.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import argparse
import logging
from urllib.parse import urlparse

# 무료로 사용 가능한 KITTI 형식 데이터 소스들
FREE_DATASETS = {
    'kitti_demo': {
        'name': 'KITTI Demo Dataset',
        'description': 'GitHub의 공개 KITTI 샘플',
        'url': 'https://github.com/utiasSTARS/pykitti/raw/master/demos/data.zip',
        'size': '~50MB',
        'samples': 10,
        'type': 'demo'
    },
    'nuscenes_sample': {
        'name': 'nuScenes Mini Sample',
        'description': 'nuScenes 공식 mini 샘플',
        'url': 'https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz',
        'size': '~1GB',
        'samples': 323,
        'type': 'nuscenes'
    },
    'waymo_sample': {
        'name': 'Waymo Sample Data',
        'description': 'Waymo 공개 샘플',
        'url': 'https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_2/individual_files',
        'size': '~100MB',
        'samples': 20,
        'type': 'waymo'
    },
    'synthetic_kitti': {
        'name': 'Synthetic KITTI',
        'description': '고품질 합성 KITTI 데이터',
        'url': 'local_generation',
        'size': 'Variable',
        'samples': 'Custom',
        'type': 'synthetic'
    }
}

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_with_progress(url, destination, chunk_size=8192):
    """진행 상황을 표시하며 파일 다운로드"""
    logger = logging.getLogger(__name__)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r다운로드 진행률: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
        
        print()  # 새 줄
        logger.info(f"✅ 다운로드 완료: {destination}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """압축 파일 해제"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📦 압축 해제 중: {archive_path}")
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        
        logger.info(f"✅ 압축 해제 완료: {extract_to}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 압축 해제 실패: {e}")
        return False

def create_advanced_synthetic_kitti(output_dir, num_samples=50):
    """고급 합성 KITTI 데이터 생성"""
    logger = logging.getLogger(__name__)
    
    logger.info("🎨 고품질 합성 KITTI 데이터 생성 중...")
    
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFilter
        import json
        
        # 출력 디렉토리 구조 생성
        splits = ['training', 'testing']
        for split in splits:
            for subdir in ['image_2', 'velodyne', 'calib']:
                os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
        
        # 실제적인 KITTI 스타일 시나리오들
        scenarios = [
            {'name': 'highway', 'bg_color': (135, 206, 235), 'road_color': (64, 64, 64)},
            {'name': 'urban', 'bg_color': (176, 196, 222), 'road_color': (48, 48, 48)},
            {'name': 'suburban', 'bg_color': (144, 238, 144), 'road_color': (70, 70, 70)},
            {'name': 'industrial', 'bg_color': (169, 169, 169), 'road_color': (52, 52, 52)},
        ]
        
        for split in splits:
            split_samples = num_samples if split == 'training' else num_samples // 2
            logger.info(f"📂 {split} 데이터 생성 중...")
            
            for i in range(split_samples):
                scenario = scenarios[i % len(scenarios)]
                
                # 1. 현실적인 이미지 생성
                img = Image.new('RGB', (1242, 375), color=scenario['bg_color'])
                draw = ImageDraw.Draw(img)
                
                # 도로 생성 (원근감 적용)
                road_width_far = 200
                road_width_near = 800
                road_y_start = 150
                road_y_end = 375
                
                # 사다리꼴 모양 도로
                road_points = [
                    (621 - road_width_far//2, road_y_start),
                    (621 + road_width_far//2, road_y_start),
                    (621 + road_width_near//2, road_y_end),
                    (621 - road_width_near//2, road_y_end)
                ]
                draw.polygon(road_points, fill=scenario['road_color'])
                
                # 차선 그리기
                for y in range(road_y_start, road_y_end, 30):
                    line_width = road_width_far + (road_width_near - road_width_far) * (y - road_y_start) / (road_y_end - road_y_start)
                    line_x_left = 621 - line_width // 6
                    line_x_right = 621 + line_width // 6
                    draw.rectangle([line_x_left, y, line_x_left + 50, y + 10], fill=(255, 255, 255))
                    draw.rectangle([line_x_right, y, line_x_right + 50, y + 10], fill=(255, 255, 255))
                
                # 차량들 추가 (다양한 크기)
                num_cars = np.random.randint(3, 8)
                for _ in range(num_cars):
                    car_x = np.random.randint(100, 1100)
                    car_y = np.random.randint(200, 350)
                    car_width = np.random.randint(80, 150)
                    car_height = np.random.randint(40, 80)
                    
                    # 차량 색상 (현실적인 색상들)
                    car_colors = [(255, 255, 255), (0, 0, 0), (128, 128, 128), (255, 0, 0), (0, 0, 255)]
                    car_color = car_colors[np.random.randint(0, len(car_colors))]
                    
                    draw.rectangle([car_x, car_y, car_x + car_width, car_y + car_height], fill=car_color)
                    
                    # 창문 추가
                    window_color = (100, 150, 200)
                    draw.rectangle([car_x + 10, car_y + 5, car_x + car_width - 10, car_y + 20], fill=window_color)
                
                # 건물들 추가
                if scenario['name'] in ['urban', 'suburban']:
                    num_buildings = np.random.randint(5, 12)
                    for _ in range(num_buildings):
                        building_x = np.random.randint(0, 1200)
                        building_y = np.random.randint(50, 150)
                        building_width = np.random.randint(100, 200)
                        building_height = np.random.randint(80, 120)
                        
                        building_colors = [(139, 69, 19), (160, 82, 45), (210, 180, 140), (128, 128, 128)]
                        building_color = building_colors[np.random.randint(0, len(building_colors))]
                        
                        draw.rectangle([building_x, building_y, building_x + building_width, building_y + building_height], fill=building_color)
                        
                        # 창문들 추가
                        for window_y in range(building_y + 10, building_y + building_height - 10, 25):
                            for window_x in range(building_x + 10, building_x + building_width - 10, 30):
                                if np.random.random() > 0.3:  # 70% 확률로 창문
                                    draw.rectangle([window_x, window_y, window_x + 15, window_y + 15], fill=(255, 255, 0))
                
                # 환경적 효과 추가
                img_array = np.array(img)
                
                # 약간의 블러로 자연스러움 추가
                img = Image.fromarray(img_array)
                img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
                
                # 노이즈 추가
                img_array = np.array(img)
                noise = np.random.normal(0, 5, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                
                final_img = Image.fromarray(img_array)
                img_path = os.path.join(output_dir, split, 'image_2', f"{i:06d}.png")
                final_img.save(img_path)
                
                # 2. 현실적인 LiDAR 데이터 생성
                num_points = np.random.randint(15000, 20000)
                
                # 현실적인 3D 포인트 분포
                # 도로면
                road_points = num_points // 3
                road_x = np.random.normal(0, 15, road_points)
                road_y = np.random.normal(-1.8, 0.2, road_points)
                road_z = np.random.uniform(2, 100, road_points)
                
                # 차량들
                car_points = num_points // 3
                car_x = np.random.normal(0, 20, car_points)
                car_y = np.random.uniform(-1.5, 1.0, car_points)
                car_z = np.random.uniform(5, 80, car_points)
                
                # 배경 (건물, 나무 등)
                bg_points = num_points - road_points - car_points
                bg_x = np.random.normal(0, 30, bg_points)
                bg_y = np.random.uniform(-1, 5, bg_points)
                bg_z = np.random.uniform(20, 150, bg_points)
                
                # 결합
                x = np.concatenate([road_x, car_x, bg_x])
                y = np.concatenate([road_y, car_y, bg_y])
                z = np.concatenate([road_z, car_z, bg_z])
                
                # 반사도 (재질별로 다르게)
                road_intensity = np.random.uniform(0.1, 0.3, road_points)
                car_intensity = np.random.uniform(0.4, 0.8, car_points)
                bg_intensity = np.random.uniform(0.2, 0.6, bg_points)
                
                intensity = np.concatenate([road_intensity, car_intensity, bg_intensity])
                
                # LiDAR 데이터 저장
                lidar_data = np.column_stack([x, y, z, intensity]).astype(np.float32)
                lidar_path = os.path.join(output_dir, split, 'velodyne', f"{i:06d}.bin")
                lidar_data.tofile(lidar_path)
                
                # 3. Calibration 데이터
                calib_content = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""
                
                calib_path = os.path.join(output_dir, split, 'calib', f"{i:06d}.txt")
                with open(calib_path, 'w') as f:
                    f.write(calib_content)
                
                if i % 10 == 0:
                    logger.info(f"  진행률: {i+1}/{split_samples}")
            
            logger.info(f"✅ {split} 완료: {split_samples}개 샘플")
        
        # 메타데이터 저장
        metadata = {
            'dataset_name': 'Advanced Synthetic KITTI',
            'scenarios': [s['name'] for s in scenarios],
            'total_samples': num_samples + num_samples // 2,
            'created_by': 'SSD-NeRF Free Sample Generator'
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("🎉 고품질 합성 KITTI 데이터 생성 완료!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 합성 데이터 생성 실패: {e}")
        return False

def show_available_free_datasets():
    """사용 가능한 무료 데이터셋 목록 표시"""
    print("\n" + "="*80)
    print("🆓 무료 사용 가능한 데이터셋")
    print("="*80)
    
    for key, info in FREE_DATASETS.items():
        print(f"\n📊 {key.upper()}")
        print(f"  • 이름: {info['name']}")
        print(f"  • 설명: {info['description']}")
        print(f"  • 크기: {info['size']}")
        print(f"  • 샘플: {info['samples']}")
        print(f"  • 타입: {info['type']}")
    
    print("\n💡 사용법:")
    print("  python scripts/download_free_samples.py --dataset synthetic_kitti")
    print("  python scripts/download_free_samples.py --dataset kitti_demo")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="🆓 무료 KITTI 형식 데이터 다운로더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 사용 가능한 데이터셋 목록 보기
  python scripts/download_free_samples.py --list

  # 고품질 합성 데이터 생성 (추천)
  python scripts/download_free_samples.py --dataset synthetic_kitti --num_samples 50

  # 공개 KITTI 샘플 다운로드
  python scripts/download_free_samples.py --dataset kitti_demo

  # 모든 무료 데이터 가져오기
  python scripts/download_free_samples.py --dataset all
        """
    )
    
    parser.add_argument('--dataset', type=str, 
                        choices=list(FREE_DATASETS.keys()) + ['all'],
                        help="다운로드할 데이터셋")
    parser.add_argument('--output_dir', type=str, default='data/free_kitti',
                        help="출력 디렉토리")
    parser.add_argument('--num_samples', type=int, default=50,
                        help="합성 데이터 샘플 수")
    parser.add_argument('--list', action='store_true',
                        help="사용 가능한 데이터셋 목록 표시")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if args.list:
        show_available_free_datasets()
        return 0
    
    if not args.dataset:
        logger.error("❌ --dataset 또는 --list 옵션이 필요합니다")
        return 1
    
    try:
        if args.dataset == 'synthetic_kitti':
            # 고품질 합성 데이터 생성
            logger.info("🎨 고품질 합성 KITTI 데이터 생성 중...")
            if create_advanced_synthetic_kitti(args.output_dir, args.num_samples):
                logger.info(f"✅ 생성 완료: {args.output_dir}")
            else:
                return 1
        
        elif args.dataset == 'all':
            # 모든 가능한 무료 데이터 가져오기
            logger.info("🆓 모든 무료 데이터 가져오기...")
            create_advanced_synthetic_kitti(args.output_dir, args.num_samples)
            # 여기에 다른 무료 소스들도 추가 가능
        
        else:
            logger.warning("⚠️ 해당 데이터셋은 아직 구현되지 않았습니다")
            logger.info("💡 대신 synthetic_kitti를 사용해보세요")
            return 1
        
        logger.info("\n💻 다음 단계:")
        logger.info(f"python scripts/run_evaluation.py --config configs/default_config.py --dynamic_checkpoint output/checkpoints/model_epoch_100.pth --model_type dynamic --data_path {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 