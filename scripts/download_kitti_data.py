#!/usr/bin/env python3
"""
KITTI 데이터셋 자동 다운로드 스크립트
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


KITTI_URLS = {
    'images_train': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip',
    'images_test': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_3.zip',
    'velodyne_train': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip',
    'calib': 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip',
}

def download_file(url, destination):


    print(f"다운로드 중: {os.path.basename(destination)}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))
    
    print(f"완료: {destination}")

def extract_zip(zip_path, extract_to):

    print(f"압축 해제 중: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"압축 해제 완료: {extract_to}")

def setup_kitti_directories(base_path):

    directories = [
        'object/training/image_2',
        'object/training/velodyne', 
        'object/training/calib',
        'object/testing/image_2',
        'object/testing/velodyne',
        'object/testing/calib'
    ]
    
    for dir_path in directories:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"디렉토리 생성: {full_path}")

def download_kitti_dataset(data_dir='data/kitti', download_all=False):
    """
    KITTI 데이터셋을 다운로드하고 설정
    
    Args:
        data_dir (str): 데이터를 저장할 디렉토리
        download_all (bool): 모든 데이터를 다운로드할지 여부 (기본: 샘플만)
    """

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    temp_dir = os.path.join(data_dir, 'temp')
    Path(temp_dir).mkdir(exist_ok=True)

    setup_kitti_directories(data_dir)
    
    print("=== KITTI 데이터셋 다운로드 ===")
    print(f"저장 경로: {os.path.abspath(data_dir)}")
    print()
    
    if not download_all:
        print("⚠️  주의: 전체 KITTI 데이터셋은 약 50GB입니다.")
        print("테스트를 위해 샘플 데이터만 다운로드하는 것을 권장합니다.")
        print("전체 데이터셋을 다운로드하려면 --all 옵션을 사용하세요.")
        print()
        
        # 샘플 데이터 다운로드 (첫 100개 이미지만)
        print("샘플 데이터를 생성하는 중...")
        create_sample_data(data_dir)
        return
    
    try:
        # 1. 캘리브레이션 데이터 다운로드 (가장 작음)
        calib_zip = os.path.join(temp_dir, 'calib.zip')
        download_file(KITTI_URLS['calib'], calib_zip)
        extract_zip(calib_zip, data_dir)
        
        # 2. 이미지 데이터 다운로드
        print("\n=== 이미지 데이터 다운로드 ===")
        images_zip = os.path.join(temp_dir, 'images.zip')
        download_file(KITTI_URLS['images_train'], images_zip)
        extract_zip(images_zip, data_dir)
        
        # 3. LiDAR 데이터 다운로드 (가장 큼)
        print("\n=== LiDAR 데이터 다운로드 ===")
        velodyne_zip = os.path.join(temp_dir, 'velodyne.zip')
        download_file(KITTI_URLS['velodyne_train'], velodyne_zip)
        extract_zip(velodyne_zip, data_dir)
        
        print("\n✅ KITTI 데이터셋 다운로드 완료!")
        
    except Exception as e:
        print(f"❌ 다운로드 중 오류 발생: {e}")
        print("수동 다운로드를 권장합니다:")
        print("1. KITTI 웹사이트 방문: http://www.cvlibs.net/datasets/kitti/")
        print("2. 계정 등록 후 데이터 다운로드")
        
    finally:
        # 임시 파일 정리
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"임시 파일 정리 완료: {temp_dir}")

def create_sample_data(data_dir):
    """
    테스트용 샘플 데이터 생성
    """
    import numpy as np
    from PIL import Image
    
    sample_dir = os.path.join(data_dir, 'object/training')
    
    # 샘플 이미지 생성 (5개)
    image_dir = os.path.join(sample_dir, 'image_2')
    for i in range(5):
        # 더미 이미지 생성 (KITTI 크기: 375x1242)
        image = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        img_pil = Image.fromarray(image)
        img_path = os.path.join(image_dir, f"{i:06d}.png")
        img_pil.save(img_path)
    
    # 샘플 LiDAR 데이터 생성
    velodyne_dir = os.path.join(sample_dir, 'velodyne')
    for i in range(5):
        # 더미 포인트 클라우드 생성 (x, y, z, intensity)
        points = np.random.randn(10000, 4).astype(np.float32)
        points[:, 3] = np.abs(points[:, 3])  # intensity는 양수
        
        lidar_path = os.path.join(velodyne_dir, f"{i:06d}.bin")
        points.tofile(lidar_path)
    
    # 샘플 캘리브레이션 데이터 생성
    calib_dir = os.path.join(sample_dir, 'calib')
    for i in range(5):
        calib_content = """P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-02 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.199936e-02 0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03
R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01
Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 -4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 -7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 -2.717806e-01
Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01"""
        
        calib_path = os.path.join(calib_dir, f"{i:06d}.txt")
        with open(calib_path, 'w') as f:
            f.write(calib_content)
    
    print(f"✅ 샘플 데이터 생성 완료: {sample_dir}")
    print("  - 5개의 더미 이미지")
    print("  - 5개의 더미 LiDAR 포인트 클라우드")
    print("  - 5개의 더미 캘리브레이션 파일")

def main():
    parser = argparse.ArgumentParser(description="KITTI 데이터셋 다운로드")
    parser.add_argument("--data-dir", default="data/kitti", help="데이터 저장 경로")
    parser.add_argument("--all", action="store_true", help="전체 데이터셋 다운로드 (약 50GB)")
    parser.add_argument("--sample", action="store_true", help="샘플 데이터만 생성")
    
    args = parser.parse_args()
    
    if args.sample:
        # 샘플 데이터만 생성
        Path(args.data_dir).mkdir(parents=True, exist_ok=True)
        setup_kitti_directories(args.data_dir)
        create_sample_data(args.data_dir)
    else:
        # 실제 데이터 다운로드
        download_kitti_dataset(args.data_dir, args.all)

if __name__ == "__main__":
    main() 