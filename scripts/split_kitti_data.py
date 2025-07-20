#!/usr/bin/env python3
"""
KITTI 데이터 분할 스크립트
다운로드한 KITTI 데이터를 훈련/검증/테스트 세트로 분할합니다.
"""

import os
import sys
import shutil
import random
import argparse
import logging
from pathlib import Path
import json

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def split_kitti_data(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, strategy='random'):
    """
    KITTI 데이터를 훈련/검증/테스트로 분할
    
    Args:
        source_dir (str): 원본 KITTI 데이터 디렉토리
        output_dir (str): 분할된 데이터를 저장할 디렉토리
        train_ratio (float): 훈련 데이터 비율
        val_ratio (float): 검증 데이터 비율  
        test_ratio (float): 테스트 데이터 비율
        strategy (str): 분할 전략 ('random', 'sequential', 'temporal')
    """
    logger = logging.getLogger(__name__)
    
    # 비율 검증
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    logger.info(f"📊 데이터 분할 시작")
    logger.info(f"📁 소스: {source_dir}")
    logger.info(f"📁 출력: {output_dir}")
    logger.info(f"📈 비율: Train {train_ratio:.1%}, Val {val_ratio:.1%}, Test {test_ratio:.1%}")
    logger.info(f"🎯 전략: {strategy}")
    
    # 이미지 파일 리스트 수집
    image_dir = os.path.join(source_dir, 'image_2')
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    total_samples = len(image_files)
    
    logger.info(f"📊 총 샘플 수: {total_samples}")
    
    if total_samples == 0:
        raise ValueError("이미지 파일을 찾을 수 없습니다")
    
    # 분할 전략에 따른 인덱스 생성
    if strategy == 'random':
        indices = list(range(total_samples))
        random.shuffle(indices)
    elif strategy == 'sequential':
        indices = list(range(total_samples))
    elif strategy == 'temporal':
        # 파일명 기준 정렬 (시간순 가정)
        indices = list(range(total_samples))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 분할 지점 계산
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    splits = {
        'training': indices[:train_end],
        'validation': indices[train_end:val_end],  
        'testing': indices[val_end:]
    }
    
    logger.info(f"📊 분할 결과:")
    for split_name, split_indices in splits.items():
        logger.info(f"  {split_name}: {len(split_indices)}개 샘플")
    
    # 출력 디렉토리 구조 생성
    for split_name in splits.keys():
        for data_type in ['image_2', 'velodyne', 'calib']:
            split_dir = os.path.join(output_dir, split_name, data_type)
            os.makedirs(split_dir, exist_ok=True)
    
    # 데이터 복사
    data_types = ['image_2', 'velodyne', 'calib']
    extensions = ['.png', '.bin', '.txt']
    
    for split_name, split_indices in splits.items():
        logger.info(f"📂 {split_name} 데이터 복사 중...")
        
        for i, sample_idx in enumerate(split_indices):
            if i % 50 == 0:
                logger.info(f"  진행률: {i+1}/{len(split_indices)}")
            
            sample_name = image_files[sample_idx]
            base_name = os.path.splitext(sample_name)[0]
            
            # 각 데이터 타입별 파일 복사
            for data_type, ext in zip(data_types, extensions):
                source_file = os.path.join(source_dir, data_type, base_name + ext)
                target_file = os.path.join(output_dir, split_name, data_type, f"{i:06d}{ext}")
                
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                else:
                    logger.warning(f"⚠️ 파일 없음: {source_file}")
        
        logger.info(f"✅ {split_name} 완료: {len(split_indices)}개 샘플")
    
    # 분할 정보 저장
    split_info = {
        'total_samples': total_samples,
        'strategy': strategy,
        'ratios': {
            'train': train_ratio,
            'validation': val_ratio,
            'test': test_ratio
        },
        'counts': {split_name: len(indices) for split_name, indices in splits.items()},
        'file_mapping': {
            split_name: [image_files[idx] for idx in indices] 
            for split_name, indices in splits.items()
        }
    }
    
    info_file = os.path.join(output_dir, 'split_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 분할 정보 저장: {info_file}")
    logger.info("🎉 데이터 분할 완료!")

def create_config_files(output_dir):
    """분할된 데이터에 맞는 config 파일들 생성"""
    logger = logging.getLogger(__name__)
    
    logger.info("⚙️ Config 파일 생성 중...")
    
    configs = {
        'train_config.py': 'training',
        'val_config.py': 'validation', 
        'test_config.py': 'testing'
    }
    
    for config_name, split_name in configs.items():
        config_content = f'''# {split_name.title()} Configuration for Split KITTI Data

config = {{
    'data': {{
        'path': '{output_dir.replace(os.sep, "/")}',
        'split': '{split_name}',
        'batch_size': 1,
        'num_workers': 4,
        'image_size': [375, 1242],
        'lidar_points': 8192,
        'focal_length': 721.5377,
    }},
    'renderer': {{
        'n_samples': 128,
        'near': 0.5,
        'far': 200.0,
    }},
    'model': {{
        'type': 'dynamic',
        'diffusion': {{
            'time_steps': 1000,
            'feature_dim': 128,
        }},
        'nerf': {{
            'embedding_dim': 256,
            'num_layers': 8,
            'use_viewdirs': True,
        }},
        'ssd_nerf': {{
            'input_dim': 3,
            'output_dim': 4,
            'num_classes': 3,
        }}
    }},
    'training': {{
        'learning_rate': 1e-4,
        'epochs': 100,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_step_size': 30,
        'scheduler_gamma': 0.1,
        'checkpoint_dir': 'output/checkpoints',
        'checkpoint_interval': 10,
        'log_dir': 'output/logs',
        'num_train_rays': 1024,
        'use_disparity_sampling': True,
        'use_ndc': False,
    }}
}}
'''
        
        config_path = os.path.join('configs', config_name)
        os.makedirs('configs', exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        logger.info(f"✅ {config_name} 생성")
    
    logger.info("⚙️ Config 파일 생성 완료")

def main():
    parser = argparse.ArgumentParser(
        description="📊 KITTI 데이터 분할 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 분할 (70% 훈련, 15% 검증, 15% 테스트)
  python scripts/split_kitti_data.py --source downloads/kitti_training --output data/kitti_split

  # 커스텀 비율로 분할
  python scripts/split_kitti_data.py --source downloads/kitti_training --output data/kitti_split --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

  # 시간순 분할 (랜덤 대신)
  python scripts/split_kitti_data.py --source downloads/kitti_training --output data/kitti_split --strategy temporal

분할 전략:
  - random: 무작위 섞기 (기본값)
  - sequential: 순차적 분할
  - temporal: 시간순 분할 (파일명 기준)
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                        help="원본 KITTI 데이터 디렉토리")
    parser.add_argument('--output', type=str, required=True,
                        help="분할된 데이터 출력 디렉토리")
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help="훈련 데이터 비율 (기본: 0.7)")
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help="검증 데이터 비율 (기본: 0.15)")
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help="테스트 데이터 비율 (기본: 0.15)")
    parser.add_argument('--strategy', type=str, choices=['random', 'sequential', 'temporal'],
                        default='random', help="분할 전략 (기본: random)")
    parser.add_argument('--seed', type=int, default=42,
                        help="랜덤 시드 (기본: 42)")
    parser.add_argument('--create_configs', action='store_true',
                        help="분할에 맞는 config 파일들 자동 생성")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # 랜덤 시드 설정
        random.seed(args.seed)
        
        # 입력 검증
        if not os.path.exists(args.source):
            logger.error(f"❌ 소스 디렉토리 없음: {args.source}")
            return 1
        
        # 데이터 분할
        split_kitti_data(
            args.source, 
            args.output,
            args.train_ratio,
            args.val_ratio, 
            args.test_ratio,
            args.strategy
        )
        
        # Config 파일 생성
        if args.create_configs:
            create_config_files(args.output)
        
        logger.info("\n💻 사용 방법:")
        logger.info("# 훈련:")
        logger.info("python scripts/run_training.py --config configs/train_config.py")
        logger.info("# 검증:")
        logger.info("python scripts/run_evaluation.py --config configs/val_config.py --dynamic_checkpoint output/checkpoints/model.pth --model_type dynamic")
        logger.info("# 테스트:")  
        logger.info("python scripts/run_evaluation.py --config configs/test_config.py --dynamic_checkpoint output/checkpoints/model.pth --model_type dynamic")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 