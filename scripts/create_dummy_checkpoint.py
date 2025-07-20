#!/usr/bin/env python3
"""
테스트용 더미 체크포인트 생성 스크립트
실제 모델 훈련 없이 evaluation을 테스트할 수 있게 해줍니다.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ssd_nerf import DynamicSSDNeRF
from src.model_arch.ssd_nerf_model import StaticSSDNeRF
from src.utils.config_utils import load_config

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_dummy_dynamic_checkpoint(config, output_path):
    """Dynamic SSD-NeRF 더미 체크포인트 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Dynamic SSD-NeRF 더미 모델 생성 중...")
        
        # 모델 생성
        model = DynamicSSDNeRF(config)
        
        # 더미 훈련 상태 생성
        dummy_checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,  # 평가에는 필요 없음
            'scheduler_state_dict': None,
            'loss': 0.5,
            'config': config,
            'model_type': 'dynamic',
            'created_by': 'dummy_checkpoint_generator',
        }
        
        # 체크포인트 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(dummy_checkpoint, output_path)
        
        logger.info(f"✅ Dynamic 더미 체크포인트 생성 완료: {output_path}")
        logger.info(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic 체크포인트 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def create_dummy_static_checkpoint(config, output_path):
    """Static SSD-NeRF 더미 체크포인트 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("📷 Static SSD-NeRF 더미 모델 생성 중...")
        
        # 모델 생성
        num_classes = config.get('model', {}).get('ssd_nerf', {}).get('num_classes', 8)
        model = StaticSSDNeRF(num_classes=num_classes)
        
        # 더미 훈련 상태 생성
        dummy_checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': None,  # 평가에는 필요 없음
            'scheduler_state_dict': None,
            'loss': 0.3,
            'config': config,
            'model_type': 'static',
            'created_by': 'dummy_checkpoint_generator',
        }
        
        # 체크포인트 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(dummy_checkpoint, output_path)
        
        logger.info(f"✅ Static 더미 체크포인트 생성 완료: {output_path}")
        logger.info(f"📊 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Static 체크포인트 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(
        description="🎭 테스트용 더미 체크포인트 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 더미 체크포인트 생성
  python scripts/create_dummy_checkpoint.py --config configs/default_config.py

  # Dynamic 모델만
  python scripts/create_dummy_checkpoint.py --config configs/default_config.py --model_type dynamic

  # Static 모델만  
  python scripts/create_dummy_checkpoint.py --config configs/default_config.py --model_type static

  # 커스텀 경로로 생성
  python scripts/create_dummy_checkpoint.py --config configs/default_config.py --output_dir output/test_checkpoints
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help="설정 파일 경로")
    parser.add_argument('--model_type', type=str, choices=['dynamic', 'static', 'both'], 
                        default='both',
                        help="생성할 모델 타입")
    parser.add_argument('--output_dir', type=str, default='output/checkpoints',
                        help="체크포인트 저장 디렉토리")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # 설정 로드
        logger.info(f"📋 설정 파일 로드: {args.config}")
        if not os.path.exists(args.config):
            logger.error(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
            return 1
            
        config = load_config(args.config)
        logger.info("✅ 설정 로드 완료")
        
        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"📁 출력 디렉토리: {args.output_dir}")
        
        success_count = 0
        
        # Dynamic 모델 생성
        if args.model_type in ['dynamic', 'both']:
            dynamic_path = os.path.join(args.output_dir, 'dummy_dynamic_model.pth')
            if create_dummy_dynamic_checkpoint(config, dynamic_path):
                success_count += 1
                logger.info(f"💾 Dynamic 체크포인트: {dynamic_path}")
        
        # Static 모델 생성
        if args.model_type in ['static', 'both']:
            static_path = os.path.join(args.output_dir, 'dummy_static_model.pth')
            if create_dummy_static_checkpoint(config, static_path):
                success_count += 1
                logger.info(f"💾 Static 체크포인트: {static_path}")
        
        if success_count > 0:
            logger.info(f"\n🎉 더미 체크포인트 생성 완료! ({success_count}개)")
            logger.info("\n💻 사용 방법:")
            
            if args.model_type in ['dynamic', 'both']:
                logger.info("# Dynamic 모델 평가:")
                logger.info(f"python scripts/run_evaluation.py --config {args.config} --dynamic_checkpoint {os.path.join(args.output_dir, 'dummy_dynamic_model.pth')} --model_type dynamic")
            
            if args.model_type in ['static', 'both']:
                logger.info("# Static 모델 평가:")
                logger.info(f"python scripts/run_evaluation.py --config {args.config} --static_checkpoint {os.path.join(args.output_dir, 'dummy_static_model.pth')} --model_type static")
            
            if args.model_type == 'both':
                logger.info("# 두 모델 비교:")
                logger.info(f"python scripts/run_evaluation.py --config {args.config} --dynamic_checkpoint {os.path.join(args.output_dir, 'dummy_dynamic_model.pth')} --static_checkpoint {os.path.join(args.output_dir, 'dummy_static_model.pth')} --model_type both")
            
            logger.info("\n🌍 다양한 환경에서 테스트:")
            logger.info("python scripts/download_other_datasets.py --dataset nuscenes_mini --num_samples 10")
            logger.info(f"python scripts/run_evaluation.py --config {args.config} --dynamic_checkpoint {os.path.join(args.output_dir, 'dummy_dynamic_model.pth')} --model_type dynamic --data_path data/other_datasets/nuscenes_mini")
            
            return 0
        else:
            logger.error("❌ 모든 체크포인트 생성에 실패했습니다.")
            return 1
    
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 