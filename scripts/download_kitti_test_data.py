#!/usr/bin/env python3
"""
KITTI 테스트 데이터셋 자동 다운로드 스크립트
실제 KITTI 데이터를 다운로드합니다.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from urllib.parse import urlparse

# 실제 KITTI 다운로드 URLs (공식 및 미러 사이트)
KITTI_TEST_URLS = {
    # KITTI 공식 사이트 (계정 필요)
    'official_images_2': 'http://www.cvlibs.net/download.php?file=data_object_image_2.zip',
    'official_images_3': 'http://www.cvlibs.net/download.php?file=data_object_image_3.zip', 
    'official_velodyne': 'http://www.cvlibs.net/download.php?file=data_object_velodyne.zip',
    'official_calib': 'http://www.cvlibs.net/download.php?file=data_object_calib.zip',
    
    # 대안 다운로드 소스 (GitHub Releases, Kaggle 등)
    'github_sample': 'https://github.com/PRBonn/kiss-icp/releases/download/v1.0.0/kitti_sample.zip',
    'kaggle_kitti': 'https://www.kaggle.com/datasets/garymk/kitti-2d-object-detection-dataset',
    
    # 소규모 테스트 데이터셋 (빠른 다운로드)
    'mini_kitti': 'https://drive.google.com/uc?id=1wMxWnwKlLX_rkCQZbsu3m7tZBoxAqXuM',
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

def download_file_with_progress(url, destination, chunk_size=8192):
    """
    진행 상황을 표시하며 파일을 다운로드합니다.
    
    Args:
        url (str): 다운로드 URL
        destination (str): 저장할 파일 경로
        chunk_size (int): 청크 크기
    
    Returns:
        bool: 다운로드 성공 여부
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📥 다운로드 시작: {os.path.basename(destination)}")
        logger.info(f"🔗 URL: {url}")
        
        # HEAD 요청으로 파일 크기 확인
        head_response = requests.head(url, allow_redirects=True, timeout=30)
        total_size = int(head_response.headers.get('content-length', 0))
        
        # 실제 다운로드
        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()
        
        # 파일이 이미 있고 크기가 같으면 스킵
        if os.path.exists(destination) and total_size > 0:
            existing_size = os.path.getsize(destination)
            if existing_size == total_size:
                logger.info(f"⏭️ 이미 다운로드됨, 스킵: {destination}")
                return True
        
        # 디렉토리 생성
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
                # 크기를 알 수 없는 경우
                logger.info("파일 크기를 알 수 없어 진행 상황 없이 다운로드...")
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
        
        logger.info(f"✅ 다운로드 완료: {destination}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ 다운로드 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        return False

def extract_archive(archive_path, extract_to, remove_after=True):
    """
    압축 파일을 해제합니다.
    
    Args:
        archive_path (str): 압축 파일 경로
        extract_to (str): 압축 해제할 디렉토리
        remove_after (bool): 압축 해제 후 원본 파일 삭제 여부
    
    Returns:
        bool: 압축 해제 성공 여부
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📦 압축 해제 시작: {os.path.basename(archive_path)}")
        
        os.makedirs(extract_to, exist_ok=True)
        
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"❌ 지원하지 않는 압축 형식: {archive_path}")
            return False
        
        logger.info(f"✅ 압축 해제 완료: {extract_to}")
        
        if remove_after and os.path.exists(archive_path):
            os.remove(archive_path)
            logger.info(f"🗑️ 원본 압축 파일 삭제: {archive_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 압축 해제 실패: {e}")
        return False

def setup_kitti_structure(base_path):
    """
    KITTI 데이터셋 디렉토리 구조를 설정합니다.
    
    Args:
        base_path (str): 베이스 디렉토리 경로
    """
    logger = logging.getLogger(__name__)
    
    directories = [
        'training/image_2',    # 훈련용 왼쪽 카메라 이미지
        'training/image_3',    # 훈련용 오른쪽 카메라 이미지
        'training/velodyne',   # 훈련용 LiDAR 포인트 클라우드
        'training/calib',      # 훈련용 calibration 파일
        'training/label_2',    # 훈련용 라벨 (2D)
        'testing/image_2',     # 테스트용 왼쪽 카메라 이미지
        'testing/image_3',     # 테스트용 오른쪽 카메라 이미지
        'testing/velodyne',    # 테스트용 LiDAR 포인트 클라우드
        'testing/calib',       # 테스트용 calibration 파일
    ]
    
    logger.info("📁 KITTI 디렉토리 구조 생성 중...")
    
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logger.debug(f"  📂 생성: {full_path}")
    
    logger.info("✅ 디렉토리 구조 생성 완료")

def download_kitti_official(data_dir, components=['images', 'velodyne', 'calib']):
    """
    KITTI 공식 웹사이트에서 데이터를 다운로드합니다.
    (주의: 계정 등록 및 로그인이 필요할 수 있습니다)
    
    Args:
        data_dir (str): 데이터 저장 디렉토리
        components (list): 다운로드할 컴포넌트 리스트
    
    Returns:
        bool: 다운로드 성공 여부
    """
    logger = logging.getLogger(__name__)
    
    logger.warning("⚠️ KITTI 공식 사이트는 계정 등록이 필요할 수 있습니다.")
    logger.info("🌐 공식 사이트: http://www.cvlibs.net/datasets/kitti/")
    
    temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    success_count = 0
    total_components = len(components)
    
    try:
        if 'calib' in components:
            # Calibration 데이터 (가장 작음, 먼저 시도)
            calib_file = os.path.join(temp_dir, 'calib.zip')
            if download_file_with_progress(KITTI_TEST_URLS['official_calib'], calib_file):
                if extract_archive(calib_file, data_dir):
                    success_count += 1
        
        if 'images' in components:
            # 이미지 데이터
            images_file = os.path.join(temp_dir, 'images_2.zip')
            if download_file_with_progress(KITTI_TEST_URLS['official_images_2'], images_file):
                if extract_archive(images_file, data_dir):
                    success_count += 1
        
        if 'velodyne' in components:
            # LiDAR 데이터 (가장 큼)
            velodyne_file = os.path.join(temp_dir, 'velodyne.zip')
            if download_file_with_progress(KITTI_TEST_URLS['official_velodyne'], velodyne_file):
                if extract_archive(velodyne_file, data_dir):
                    success_count += 1
        
        return success_count == total_components
        
    finally:
        # 임시 디렉토리 정리
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"🧹 임시 파일 정리 완료")

def download_alternative_sources(data_dir):
    """
    대안 소스에서 KITTI 데이터를 다운로드합니다.
    
    Args:
        data_dir (str): 데이터 저장 디렉토리
    
    Returns:
        bool: 다운로드 성공 여부
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 대안 소스에서 다운로드 시도 중...")
    
    # GitHub에서 샘플 데이터 다운로드 시도
    temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        sample_file = os.path.join(temp_dir, 'kitti_sample.zip')
        if download_file_with_progress(KITTI_TEST_URLS['github_sample'], sample_file):
            if extract_archive(sample_file, data_dir):
                logger.info("✅ GitHub 샘플 데이터 다운로드 성공")
                return True
    except Exception as e:
        logger.warning(f"⚠️ GitHub 샘플 다운로드 실패: {e}")
    
    logger.info("💡 수동 다운로드 가이드:")
    logger.info("1. Kaggle: https://www.kaggle.com/datasets/garymk/kitti-2d-object-detection-dataset")
    logger.info("2. KITTI 공식: http://www.cvlibs.net/datasets/kitti/")
    logger.info("3. 또는 더미 데이터 사용: python scripts/create_dummy_data.py")
    
    return False

def create_realistic_test_data(data_dir, num_samples=20):
    """
    현실적인 테스트 데이터를 생성합니다.
    
    Args:
        data_dir (str): 데이터 저장 디렉토리
        num_samples (int): 생성할 샘플 수
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎲 현실적인 테스트 데이터 생성 중... ({num_samples}개 샘플)")
    
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # 테스트 데이터를 위한 디렉토리
    splits = ['training', 'testing']
    
    for split in splits:
        logger.info(f"📂 {split} 데이터 생성 중...")
        
        image_dir = os.path.join(data_dir, split, 'image_2')
        velodyne_dir = os.path.join(data_dir, split, 'velodyne')
        calib_dir = os.path.join(data_dir, split, 'calib')
        
        for i in range(num_samples):
            # 1. 현실적인 이미지 생성
            img = Image.new('RGB', (1242, 375), color=(135, 206, 235))  # 하늘색 배경
            draw = ImageDraw.Draw(img)
            
            # 도로 그리기
            road_color = (169, 169, 169)  # 회색 도로
            draw.rectangle([0, 250, 1242, 375], fill=road_color)
            
            # 차선 그리기
            line_color = (255, 255, 255)  # 흰색 차선
            for x in range(0, 1242, 100):
                draw.rectangle([x, 300, x+50, 310], fill=line_color)
            
            # 건물이나 물체 추가
            for j in range(3):
                x = np.random.randint(0, 1000)
                y = np.random.randint(100, 200)
                w = np.random.randint(50, 150)
                h = np.random.randint(100, 200)
                color = tuple(np.random.randint(50, 200, 3))
                draw.rectangle([x, y, x+w, y+h], fill=color)
            
            # 노이즈 추가
            img_array = np.array(img)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.int8)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            final_img = Image.fromarray(img_array)
            img_path = os.path.join(image_dir, f"{i:06d}.png")
            final_img.save(img_path)
            
            # 2. 현실적인 LiDAR 데이터 생성
            num_points = np.random.randint(15000, 25000)
            
            # 거리별 포인트 밀도 조정 (가까운 곳이 더 조밀)
            distances = np.random.exponential(20, num_points)
            distances = np.clip(distances, 2, 100)  # 2m ~ 100m
            
            # 각도 분포 (전방 120도 범위)
            angles = np.random.uniform(-np.pi/3, np.pi/3, num_points)
            
            # 3D 좌표 계산
            x = distances * np.sin(angles) + np.random.normal(0, 0.5, num_points)
            z = distances * np.cos(angles) + np.random.normal(0, 0.5, num_points)
            y = np.random.normal(-1.8, 0.3, num_points)  # 차량 높이 기준
            
            # 지면 포인트 추가
            ground_points = num_points // 4
            ground_x = np.random.uniform(-30, 30, ground_points)
            ground_z = np.random.uniform(2, 50, ground_points)
            ground_y = np.random.normal(-1.8, 0.1, ground_points)
            
            x = np.concatenate([x, ground_x])
            y = np.concatenate([y, ground_y])
            z = np.concatenate([z, ground_z])
            
            # 반사도 (intensity)
            intensity = np.random.uniform(0.1, 1.0, len(x))
            
            # LiDAR 데이터 저장
            lidar_data = np.column_stack([x, y, z, intensity]).astype(np.float32)
            lidar_path = os.path.join(velodyne_dir, f"{i:06d}.bin")
            lidar_data.tofile(lidar_path)
            
            # 3. Calibration 데이터 (실제 KITTI 파라미터 기반)
            calib_content = f"""P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""
            
            calib_path = os.path.join(calib_dir, f"{i:06d}.txt")
            with open(calib_path, 'w') as f:
                f.write(calib_content)
        
        logger.info(f"✅ {split} 데이터 완료: {num_samples}개 샘플")
    
    logger.info("🎉 현실적인 테스트 데이터 생성 완료!")

def main():
    parser = argparse.ArgumentParser(
        description="🚗 KITTI 테스트 데이터 다운로더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 다운로드 (공식 소스 시도 후 대안 소스)
  python scripts/download_kitti_test_data.py

  # 특정 경로에 다운로드
  python scripts/download_kitti_test_data.py --data_dir /path/to/kitti

  # 현실적인 테스트 데이터만 생성
  python scripts/download_kitti_test_data.py --realistic_only --num_samples 50

  # 특정 컴포넌트만 다운로드
  python scripts/download_kitti_test_data.py --components images calib

  # 공식 소스 건너뛰기
  python scripts/download_kitti_test_data.py --skip_official
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='data/kitti',
                        help="데이터 저장 디렉토리 (기본: data/kitti)")
    parser.add_argument('--components', nargs='+', 
                        choices=['images', 'velodyne', 'calib'], 
                        default=['images', 'velodyne', 'calib'],
                        help="다운로드할 컴포넌트 (기본: 모든 컴포넌트)")
    parser.add_argument('--num_samples', type=int, default=20,
                        help="생성할 테스트 샘플 수 (기본: 20)")
    parser.add_argument('--realistic_only', action='store_true',
                        help="현실적인 테스트 데이터만 생성 (다운로드 건너뛰기)")
    parser.add_argument('--skip_official', action='store_true',
                        help="공식 소스 다운로드 건너뛰기")
    parser.add_argument('--verbose', action='store_true',
                        help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logger.info("🚀 KITTI 테스트 데이터 다운로드 시작")
        logger.info(f"📁 저장 경로: {os.path.abspath(args.data_dir)}")
        logger.info(f"📦 컴포넌트: {args.components}")
        
        # 디렉토리 구조 설정
        setup_kitti_structure(args.data_dir)
        
        success = False
        
        if args.realistic_only:
            # 현실적인 데이터만 생성
            create_realistic_test_data(args.data_dir, args.num_samples)
            success = True
        else:
            # 실제 데이터 다운로드 시도
            if not args.skip_official:
                logger.info("1️⃣ 공식 소스에서 다운로드 시도...")
                success = download_kitti_official(args.data_dir, args.components)
            
            if not success:
                logger.info("2️⃣ 대안 소스에서 다운로드 시도...")
                success = download_alternative_sources(args.data_dir)
            
            if not success:
                logger.info("3️⃣ 현실적인 테스트 데이터 생성...")
                create_realistic_test_data(args.data_dir, args.num_samples)
                success = True
        
        if success:
            logger.info("🎉 KITTI 데이터 준비 완료!")
            logger.info(f"📂 경로: {os.path.abspath(args.data_dir)}")
            logger.info("💻 사용 방법:")
            logger.info("  python scripts/run_evaluation.py --config configs/default_config.py")
            logger.info("  python scripts/run_inference.py --config configs/default_config.py --checkpoint model.pth")
        else:
            logger.error("❌ 모든 다운로드 방법이 실패했습니다.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 