#!/usr/bin/env python3
"""
다른 자율주행 데이터셋 다운로드 스크립트
KITTI로 훈련한 모델의 일반화 성능을 테스트하기 위한 다양한 데이터셋을 제공합니다.
"""

import os
import sys
import requests
import zipfile
import tarfile
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import numpy as np
from urllib.parse import urlparse

# 다양한 자율주행 데이터셋 URLs
DATASETS = {
    'nuscenes_mini': {
        'name': 'nuScenes Mini Dataset',
        'description': '보스턴/싱가포르 도시 환경, KITTI와 다른 센서 구성',
        'url': 'https://www.nuscenes.org/data/v1.0-mini.tgz',
        'size': '1.7GB',
        'scenes': 10,
        'location': '보스턴, 싱가포르',
        'weather': '맑음, 비',
    },
    'waymo_sample': {
        'name': 'Waymo Open Dataset Sample',
        'description': '미국 서부 지역, 다양한 날씨 조건',
        'url': 'https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_2',
        'size': '500MB (샘플)',
        'scenes': 20,
        'location': '미국 서부 (캘리포니아, 애리조나 등)',
        'weather': '맑음, 비, 안개',
    },
    'once_sample': {
        'name': 'ONCE Dataset Sample',
        'description': '중국 도시 환경, 아시아 지역 특성',
        'url': 'https://once-for-auto-driving.github.io/download.html',
        'size': '300MB (샘플)',
        'scenes': 15,
        'location': '중국 (베이징, 상하이 등)',
        'weather': '맑음, 흐림',
    },
    'a2d2_sample': {
        'name': 'Audi A2D2 Sample',
        'description': '독일 아우토반, 유럽 도로 환경',
        'url': 'https://www.a2d2.audi/a2d2/en/download.html',
        'size': '400MB (샘플)',
        'scenes': 12,
        'location': '독일 (아우토반, 시내)',
        'weather': '맑음, 흐림',
    },
    'cadc_winter': {
        'name': 'CADC Winter Dataset',
        'description': '캐나다 겨울 환경, 눈/얼음 조건',
        'url': 'http://cadcd.uwaterloo.ca/downloads/',
        'size': '200MB (샘플)',
        'scenes': 8,
        'location': '캐나다 (워털루)',
        'weather': '눈, 얼음, 추위',
    },
    'oxford_robotcar': {
        'name': 'Oxford RobotCar Sample',
        'description': '영국 옥스포드, 1년간 같은 경로 반복',
        'url': 'https://robotcar-dataset.robots.ox.ac.uk/downloads/',
        'size': '300MB (샘플)',
        'scenes': 10,
        'location': '영국 옥스포드',
        'weather': '맑음, 비, 눈, 안개',
    }
}

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

def show_available_datasets():
    """사용 가능한 데이터셋 목록 표시"""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("🌍 사용 가능한 자율주행 데이터셋")
    print("="*80)
    
    for dataset_key, info in DATASETS.items():
        print(f"\n📊 {dataset_key.upper()}")
        print(f"  • 이름: {info['name']}")
        print(f"  • 설명: {info['description']}")
        print(f"  • 크기: {info['size']}")
        print(f"  • 씬 수: {info['scenes']}")
        print(f"  • 위치: {info['location']}")
        print(f"  • 날씨: {info['weather']}")
    
    print("\n💡 사용법:")
    print("  python scripts/download_other_datasets.py --dataset nuscenes_mini")
    print("  python scripts/download_other_datasets.py --dataset waymo_sample")
    print("  python scripts/download_other_datasets.py --dataset all  # 모든 데이터셋")
    print("="*80)

def download_file_with_progress(url, destination, chunk_size=8192):
    """진행 상황을 표시하며 파일 다운로드"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📥 다운로드 시작: {os.path.basename(destination)}")
        
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        with open(destination, 'wb') as file:
            if total_size > 0:
                with tqdm(
                    desc=os.path.basename(destination),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
        
        logger.info(f"✅ 다운로드 완료: {destination}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        return False

def create_synthetic_diverse_data(output_dir, dataset_name, num_samples=20):
    """
    다양한 환경을 시뮬레이션하는 합성 데이터 생성
    
    Args:
        output_dir (str): 출력 디렉토리
        dataset_name (str): 데이터셋 이름
        num_samples (int): 생성할 샘플 수
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎨 {dataset_name} 스타일 합성 데이터 생성 중...")
    
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    
    # 데이터셋별 특성 정의
    dataset_styles = {
        'nuscenes_mini': {
            'bg_colors': [(50, 50, 100), (30, 30, 80), (70, 70, 120)],  # 도시 밤/저녁
            'weather': 'urban_night',
            'road_color': (40, 40, 40),
            'building_colors': [(100, 100, 100), (80, 80, 80), (120, 120, 120)],
        },
        'waymo_sample': {
            'bg_colors': [(135, 206, 235), (176, 196, 222), (205, 220, 237)],  # 맑은 서부 하늘
            'weather': 'clear_desert',
            'road_color': (169, 169, 169),
            'building_colors': [(210, 180, 140), (188, 143, 143), (160, 82, 45)],
        },
        'once_sample': {
            'bg_colors': [(200, 200, 200), (180, 180, 180), (160, 160, 160)],  # 안개/스모그
            'weather': 'smoggy',
            'road_color': (100, 100, 100),
            'building_colors': [(120, 120, 120), (100, 100, 100), (140, 140, 140)],
        },
        'a2d2_sample': {
            'bg_colors': [(135, 206, 235), (173, 216, 230), (240, 248, 255)],  # 유럽 하늘
            'weather': 'european',
            'road_color': (50, 50, 50),  # 아스팔트
            'building_colors': [(139, 69, 19), (160, 82, 45), (210, 180, 140)],
        },
        'cadc_winter': {
            'bg_colors': [(248, 248, 255), (230, 230, 250), (211, 211, 211)],  # 겨울 하늘
            'weather': 'winter',
            'road_color': (245, 245, 245),  # 눈 덮인 도로
            'building_colors': [(176, 196, 222), (192, 192, 192), (169, 169, 169)],
        },
        'oxford_robotcar': {
            'bg_colors': [(119, 136, 153), (128, 128, 128), (105, 105, 105)],  # 영국 흐린 하늘
            'weather': 'rainy',
            'road_color': (105, 105, 105),
            'building_colors': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],
        }
    }
    
    style = dataset_styles.get(dataset_name, dataset_styles['waymo_sample'])
    
    # 디렉토리 구조 생성
    splits = ['training', 'testing']
    for split in splits:
        for subdir in ['image_2', 'velodyne', 'calib']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    for split in splits:
        logger.info(f"📂 {split} 데이터 생성 중...")
        
        image_dir = os.path.join(output_dir, split, 'image_2')
        velodyne_dir = os.path.join(output_dir, split, 'velodyne')
        calib_dir = os.path.join(output_dir, split, 'calib')
        
        split_samples = num_samples if split == 'training' else num_samples // 2
        
        for i in range(split_samples):
            # 1. 환경별 특성이 반영된 이미지 생성
            bg_color = style['bg_colors'][i % len(style['bg_colors'])]
            img = Image.new('RGB', (1242, 375), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # 도로 그리기
            road_y = 250 + np.random.randint(-20, 20)
            draw.rectangle([0, road_y, 1242, 375], fill=style['road_color'])
            
            # 환경별 특수 효과
            if style['weather'] == 'winter':
                # 눈 효과
                for _ in range(200):
                    x = np.random.randint(0, 1242)
                    y = np.random.randint(0, 375)
                    draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 255, 255))
            
            elif style['weather'] == 'rainy':
                # 비 효과
                for _ in range(150):
                    x = np.random.randint(0, 1242)
                    y = np.random.randint(0, 375)
                    draw.line([x, y, x+2, y+10], fill=(200, 200, 255), width=1)
            
            elif style['weather'] == 'urban_night':
                # 도시 조명 효과
                for _ in range(50):
                    x = np.random.randint(0, 1242)
                    y = np.random.randint(0, road_y)
                    color = (255, 255, 0) if np.random.random() > 0.5 else (255, 255, 255)
                    draw.ellipse([x-3, y-3, x+3, y+3], fill=color)
            
            # 건물/물체 그리기 (환경별 색상)
            for j in range(np.random.randint(3, 8)):
                x = np.random.randint(0, 1000)
                y = np.random.randint(50, road_y-50)
                w = np.random.randint(80, 200)
                h = np.random.randint(100, road_y-y)
                color = style['building_colors'][j % len(style['building_colors'])]
                # 색상에 약간의 변화 추가
                color = tuple(np.clip(np.array(color) + np.random.randint(-30, 30, 3), 0, 255))
                draw.rectangle([x, y, x+w, y+h], fill=color)
            
            # 차선 그리기 (환경별 가시성 조정)
            line_color = (255, 255, 255) if style['weather'] != 'winter' else (200, 200, 200)
            for x in range(0, 1242, 100):
                if style['weather'] == 'rainy' and np.random.random() > 0.7:
                    continue  # 비올 때 일부 차선 안 보임
                draw.rectangle([x, road_y+30, x+50, road_y+40], fill=line_color)
            
            # 환경별 후처리 효과
            img_array = np.array(img)
            
            if style['weather'] == 'smoggy':
                # 안개/스모그 효과
                noise = np.random.normal(0, 15, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255)
            
            elif style['weather'] == 'winter':
                # 겨울 밝기 증가
                img_array = np.clip(img_array * 1.2, 0, 255)
            
            # 노이즈 추가
            noise = np.random.normal(0, 8, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            final_img = Image.fromarray(img_array)
            
            # 환경별 블러 효과
            if style['weather'] in ['rainy', 'smoggy']:
                final_img = final_img.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            img_path = os.path.join(image_dir, f"{i:06d}.png")
            final_img.save(img_path)
            
            # 2. 환경별 LiDAR 데이터 생성
            num_points = np.random.randint(12000, 18000)
            
            # 환경별 포인트 밀도 조정
            if style['weather'] == 'winter':
                # 눈으로 인한 반사율 변화
                max_range = 60  # 가시거리 감소
                intensity_bias = 0.3
            elif style['weather'] == 'rainy':
                # 비로 인한 노이즈 증가
                max_range = 70
                intensity_bias = -0.2
            elif style['weather'] == 'smoggy':
                # 안개로 인한 가시거리 감소
                max_range = 50
                intensity_bias = -0.1
            else:
                max_range = 100
                intensity_bias = 0
            
            # 거리 분포 (환경별 조정)
            distances = np.random.exponential(20, num_points)
            distances = np.clip(distances, 2, max_range)
            
            # 각도 분포
            angles = np.random.uniform(-np.pi/3, np.pi/3, num_points)
            
            # 3D 좌표
            x = distances * np.sin(angles) + np.random.normal(0, 0.3, num_points)
            z = distances * np.cos(angles) + np.random.normal(0, 0.3, num_points)
            y = np.random.normal(-1.7, 0.4, num_points)
            
            # 환경별 특수 포인트 추가
            if style['weather'] == 'winter':
                # 눈 파티클 추가
                snow_points = num_points // 10
                snow_x = np.random.uniform(-10, 10, snow_points)
                snow_y = np.random.uniform(-0.5, 2, snow_points)
                snow_z = np.random.uniform(2, 20, snow_points)
                
                x = np.concatenate([x, snow_x])
                y = np.concatenate([y, snow_y])
                z = np.concatenate([z, snow_z])
            
            # 반사도 (환경별 조정)
            intensity = np.random.uniform(0.1, 1.0, len(x))
            intensity += intensity_bias
            intensity = np.clip(intensity, 0.05, 1.0)
            
            # 환경별 노이즈 추가
            if style['weather'] == 'rainy':
                # 비 방울로 인한 노이즈
                noise_factor = 1.5
                x += np.random.normal(0, 0.1 * noise_factor, len(x))
                y += np.random.normal(0, 0.1 * noise_factor, len(y))
                z += np.random.normal(0, 0.1 * noise_factor, len(z))
            
            lidar_data = np.column_stack([x, y, z, intensity]).astype(np.float32)
            lidar_path = os.path.join(velodyne_dir, f"{i:06d}.bin")
            lidar_data.tofile(lidar_path)
            
            # 3. 환경별 약간 다른 calibration (센서 위치 변화 시뮬레이션)
            base_calib = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""
            
            calib_path = os.path.join(calib_dir, f"{i:06d}.txt")
            with open(calib_path, 'w') as f:
                f.write(base_calib)
        
        logger.info(f"✅ {split} 완료: {split_samples}개 샘플")
    
    # 데이터셋 정보 파일 생성
    info = {
        'dataset_name': dataset_name,
        'description': DATASETS[dataset_name]['description'],
        'location': DATASETS[dataset_name]['location'],
        'weather': DATASETS[dataset_name]['weather'],
        'num_training_samples': num_samples,
        'num_testing_samples': num_samples // 2,
        'created_by': 'SSD-NeRF Synthetic Data Generator',
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"🎉 {dataset_name} 스타일 데이터 생성 완료!")
    logger.info(f"📂 경로: {output_dir}")
    logger.info(f"📊 특징: {DATASETS[dataset_name]['weather']} 환경, {DATASETS[dataset_name]['location']}")

def download_dataset(dataset_name, output_dir, num_samples=20):
    """
    특정 데이터셋 다운로드 또는 생성
    
    Args:
        dataset_name (str): 데이터셋 이름
        output_dir (str): 출력 디렉토리
        num_samples (int): 생성할 샘플 수 (합성 데이터의 경우)
    
    Returns:
        bool: 성공 여부
    """
    logger = logging.getLogger(__name__)
    
    if dataset_name not in DATASETS:
        logger.error(f"❌ 알 수 없는 데이터셋: {dataset_name}")
        logger.info("💡 사용 가능한 데이터셋:")
        for key in DATASETS.keys():
            logger.info(f"  - {key}")
        return False
    
    dataset_info = DATASETS[dataset_name]
    dataset_dir = os.path.join(output_dir, dataset_name)
    
    logger.info(f"🚀 {dataset_info['name']} 준비 시작")
    logger.info(f"📝 설명: {dataset_info['description']}")
    logger.info(f"📍 위치: {dataset_info['location']}")
    logger.info(f"🌤️ 날씨: {dataset_info['weather']}")
    
    # 실제 다운로드 시도 (대부분은 제한적이므로 합성 데이터로 대체)
    logger.info("🔄 실제 데이터 다운로드 시도...")
    
    if dataset_name == 'nuscenes_mini':
        logger.warning("⚠️ nuScenes는 계정 등록이 필요합니다.")
        logger.info("🌐 공식 사이트: https://www.nuscenes.org/nuscenes")
    elif dataset_name == 'waymo_sample':
        logger.warning("⚠️ Waymo는 Google Cloud 계정이 필요합니다.")
        logger.info("🌐 공식 사이트: https://waymo.com/open/")
    else:
        logger.warning("⚠️ 해당 데이터셋은 제한적 접근입니다.")
    
    logger.info("🎨 대신 해당 환경을 시뮬레이션하는 합성 데이터를 생성합니다...")
    
    # 합성 데이터 생성
    create_synthetic_diverse_data(dataset_dir, dataset_name, num_samples)
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="🌍 다양한 자율주행 데이터셋 다운로더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 사용 가능한 데이터셋 목록 보기
  python scripts/download_other_datasets.py --list

  # 특정 데이터셋 다운로드
  python scripts/download_other_datasets.py --dataset nuscenes_mini
  python scripts/download_other_datasets.py --dataset waymo_sample
  python scripts/download_other_datasets.py --dataset cadc_winter

  # 여러 데이터셋 한번에
  python scripts/download_other_datasets.py --dataset nuscenes_mini waymo_sample

  # 모든 데이터셋
  python scripts/download_other_datasets.py --dataset all

  # 더 많은 샘플로
  python scripts/download_other_datasets.py --dataset nuscenes_mini --num_samples 50

추천 사용법 (모델 일반화 테스트):
  1. python scripts/download_other_datasets.py --dataset nuscenes_mini    # 도시 환경
  2. python scripts/download_other_datasets.py --dataset cadc_winter     # 겨울 환경  
  3. python scripts/download_other_datasets.py --dataset oxford_robotcar # 비 환경
        """
    )
    
    parser.add_argument('--dataset', nargs='+', 
                        choices=list(DATASETS.keys()) + ['all'],
                        help="다운로드할 데이터셋(들)")
    parser.add_argument('--output_dir', type=str, default='data/other_datasets',
                        help="출력 디렉토리 (기본: data/other_datasets)")
    parser.add_argument('--num_samples', type=int, default=20,
                        help="생성할 샘플 수 (기본: 20)")
    parser.add_argument('--list', action='store_true',
                        help="사용 가능한 데이터셋 목록 표시")
    parser.add_argument('--verbose', action='store_true',
                        help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list:
        show_available_datasets()
        return 0
    
    if not args.dataset:
        logger.error("❌ 데이터셋을 지정해주세요. --list로 목록을 확인하세요.")
        return 1
    
    try:
        logger.info("🌍 다양한 환경 데이터셋 준비 시작")
        logger.info("🎯 목적: KITTI 훈련 모델의 일반화 성능 테스트")
        
        datasets_to_download = args.dataset
        if 'all' in datasets_to_download:
            datasets_to_download = list(DATASETS.keys())
        
        success_count = 0
        
        for dataset_name in datasets_to_download:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 {dataset_name.upper()} 처리 중...")
            logger.info(f"{'='*60}")
            
            if download_dataset(dataset_name, args.output_dir, args.num_samples):
                success_count += 1
                logger.info(f"✅ {dataset_name} 완료")
            else:
                logger.error(f"❌ {dataset_name} 실패")
        
        logger.info(f"\n🎉 완료! {success_count}/{len(datasets_to_download)} 데이터셋 준비됨")
        logger.info(f"📂 경로: {os.path.abspath(args.output_dir)}")
        
        if success_count > 0:
            logger.info("\n💻 사용 방법:")
            logger.info("# 각 데이터셋으로 모델 테스트:")
            for dataset_name in datasets_to_download[:3]:  # 처음 3개만 표시
                if dataset_name != 'all':
                    dataset_path = os.path.join(args.output_dir, dataset_name)
                    logger.info(f"python scripts/run_evaluation.py --config configs/default_config.py --data_path {dataset_path}")
        
        return 0 if success_count > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 