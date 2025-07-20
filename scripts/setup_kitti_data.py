#!/usr/bin/env python3
"""
KITTI 데이터셋 설정 도우미 스크립트
다운로드한 KITTI 데이터를 올바른 구조로 배치합니다.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_kitti_structure(base_path):
    """KITTI 디렉토리 구조 생성"""
    logger = logging.getLogger(__name__)
    
    required_dirs = [
        'training/image_2',
        'training/velodyne', 
        'training/calib',
        'testing/image_2',
        'testing/velodyne',
        'testing/calib'
    ]
    
    logger.info(f"📁 KITTI 디렉토리 구조 생성: {base_path}")
    
    for dir_path in required_dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"  📂 생성: {full_path}")
    
    logger.info("✅ 디렉토리 구조 생성 완료")

def process_object_detection_data(source_dir, target_dir):
    """3D Object Detection 데이터 처리"""
    logger = logging.getLogger(__name__)
    
    logger.info("🚗 3D Object Detection 데이터 처리 중...")
    
    # training 데이터 처리
    splits = ['training', 'testing']
    data_types = ['image_2', 'velodyne', 'calib']
    
    for split in splits:
        for data_type in data_types:
            source_path = os.path.join(source_dir, split, data_type)
            target_path = os.path.join(target_dir, split, data_type)
            
            if os.path.exists(source_path):
                logger.info(f"📂 처리 중: {split}/{data_type}")
                
                # 파일 패턴에 따라 처리
                if data_type == 'image_2':
                    pattern = '*.png'
                elif data_type == 'velodyne':
                    pattern = '*.bin'
                elif data_type == 'calib':
                    pattern = '*.txt'
                
                # 파일 복사
                source_files = list(Path(source_path).glob(pattern))
                copied_count = 0
                
                for file_path in source_files:
                    target_file = os.path.join(target_path, file_path.name)
                    shutil.copy2(file_path, target_file)
                    copied_count += 1
                
                logger.info(f"  ✅ {copied_count}개 파일 복사 완료")
            else:
                logger.warning(f"⚠️ 소스 경로 없음: {source_path}")

def process_raw_data(source_dir, target_dir, sequence_name):
    """Raw Data 시퀀스 처리"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"🎬 Raw Data 시퀀스 처리: {sequence_name}")
    
    # Raw data 구조: YYYY_MM_DD/YYYY_MM_DD_drive_XXXX_sync/
    date_part = sequence_name[:10]  # 2011_09_26
    
    sequence_path = os.path.join(source_dir, date_part, f"{sequence_name}_sync")
    
    if not os.path.exists(sequence_path):
        logger.error(f"❌ 시퀀스 경로 없음: {sequence_path}")
        return False
    
    # 이미지 처리 (image_02/data/*.png)
    image_source = os.path.join(sequence_path, 'image_02', 'data')
    image_target = os.path.join(target_dir, 'training', 'image_2')
    
    if os.path.exists(image_source):
        image_files = list(Path(image_source).glob('*.png'))
        logger.info(f"📸 이미지 {len(image_files)}개 복사 중...")
        
        for i, img_file in enumerate(image_files):
            # 파일명을 6자리 숫자로 변경
            new_name = f"{i:06d}.png"
            target_file = os.path.join(image_target, new_name)
            shutil.copy2(img_file, target_file)
        
        logger.info(f"✅ 이미지 복사 완료: {len(image_files)}개")
    
    # LiDAR 처리 (velodyne_points/data/*.bin)
    lidar_source = os.path.join(sequence_path, 'velodyne_points', 'data')
    lidar_target = os.path.join(target_dir, 'training', 'velodyne')
    
    if os.path.exists(lidar_source):
        lidar_files = list(Path(lidar_source).glob('*.bin'))
        logger.info(f"📡 LiDAR {len(lidar_files)}개 복사 중...")
        
        for i, lidar_file in enumerate(lidar_files):
            new_name = f"{i:06d}.bin"
            target_file = os.path.join(lidar_target, new_name)
            shutil.copy2(lidar_file, target_file)
        
        logger.info(f"✅ LiDAR 복사 완료: {len(lidar_files)}개")
    
    # Calibration 처리 (calib_*.txt 파일들)
    calib_source = os.path.join(source_dir, date_part)
    calib_target = os.path.join(target_dir, 'training', 'calib')
    
    calib_files = list(Path(calib_source).glob('calib_*.txt'))
    if calib_files:
        logger.info(f"📏 Calibration 파일 복사 중...")
        
        # 이미지 수만큼 calibration 파일 복제
        image_count = len(list(Path(image_target).glob('*.png')))
        
        for i in range(image_count):
            # 대표 calibration 파일 사용 (보통 첫 번째)
            source_calib = calib_files[0]
            new_name = f"{i:06d}.txt"
            target_file = os.path.join(calib_target, new_name)
            shutil.copy2(source_calib, target_file)
        
        logger.info(f"✅ Calibration 복사 완료: {image_count}개")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="🛠️ KITTI 데이터셋 설정 도우미",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 디렉토리 구조만 생성
  python scripts/setup_kitti_data.py --target_dir data/kitti --create_structure_only

  # 3D Object Detection 데이터 설정
  python scripts/setup_kitti_data.py --source_dir downloads/kitti_object --target_dir data/kitti --data_type object_detection

  # Raw Data 시퀀스 설정
  python scripts/setup_kitti_data.py --source_dir downloads/kitti_raw --target_dir data/kitti --data_type raw_data --sequence 2011_09_26_drive_0001

  # 다운로드한 zip 파일 자동 처리
  python scripts/setup_kitti_data.py --zip_file downloads/data_object_image_2.zip --target_dir data/kitti
        """
    )
    
    parser.add_argument('--source_dir', type=str, 
                        help="KITTI 데이터 소스 디렉토리")
    parser.add_argument('--target_dir', type=str, default='data/kitti',
                        help="KITTI 데이터 타겟 디렉토리")
    parser.add_argument('--data_type', type=str, choices=['object_detection', 'raw_data'],
                        help="데이터 타입")
    parser.add_argument('--sequence', type=str,
                        help="Raw data 시퀀스 이름 (예: 2011_09_26_drive_0001)")
    parser.add_argument('--create_structure_only', action='store_true',
                        help="디렉토리 구조만 생성")
    parser.add_argument('--zip_file', type=str,
                        help="압축 파일 자동 처리")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # 타겟 디렉토리 구조 생성
        create_kitti_structure(args.target_dir)
        
        if args.create_structure_only:
            logger.info("🎉 디렉토리 구조 생성 완료!")
            logger.info(f"📁 경로: {os.path.abspath(args.target_dir)}")
            logger.info("\n💡 다음 단계:")
            logger.info("1. KITTI 데이터를 다운로드")
            logger.info("2. 이 스크립트로 데이터 배치")
            logger.info("3. python scripts/run_evaluation.py로 테스트")
            return 0
        
        if args.zip_file:
            logger.info(f"📦 압축 파일 처리: {args.zip_file}")
            # TODO: ZIP 파일 자동 해제 및 처리
            logger.warning("⚠️ ZIP 파일 자동 처리는 구현 예정")
            return 0
        
        if not args.source_dir:
            logger.error("❌ --source_dir 또는 --create_structure_only 필요")
            return 1
        
        if not os.path.exists(args.source_dir):
            logger.error(f"❌ 소스 디렉토리 없음: {args.source_dir}")
            return 1
        
        # 데이터 타입에 따른 처리
        if args.data_type == 'object_detection':
            process_object_detection_data(args.source_dir, args.target_dir)
        elif args.data_type == 'raw_data':
            if not args.sequence:
                logger.error("❌ Raw data 처리시 --sequence 필요")
                return 1
            process_raw_data(args.source_dir, args.target_dir, args.sequence)
        else:
            logger.error("❌ --data_type 지정 필요")
            return 1
        
        # 설치 확인
        logger.info("\n🔍 설치 확인...")
        splits = ['training', 'testing']
        
        for split in splits:
            image_path = os.path.join(args.target_dir, split, 'image_2')
            image_count = len(list(Path(image_path).glob('*.png')))
            logger.info(f"  📊 {split}: {image_count}개 이미지")
        
        logger.info("\n🎉 KITTI 데이터 설정 완료!")
        logger.info(f"📁 경로: {os.path.abspath(args.target_dir)}")
        logger.info("\n💻 다음 단계:")
        logger.info("python scripts/run_evaluation.py --config configs/default_config.py --dynamic_checkpoint output/checkpoints/model_epoch_100.pth --model_type dynamic")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 