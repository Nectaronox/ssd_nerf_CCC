#!/usr/bin/env python3
"""
SSD-NeRF Training Script
개선된 학습 실행 스크립트 with 에러 처리, 설정 검증, GPU 확인 등
"""

import argparse
import sys
import os
import signal
import logging
import time
from datetime import datetime
import torch

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.training.trainer import Trainer
    from src.utils.config_utils import load_config
except ImportError as e:
    print(f"❌ 모듈 임포트 오류: {e}")
    print("프로젝트 루트 디렉토리에서 실행했는지 확인하세요.")
    sys.exit(1)

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

def signal_handler(signum, frame):
    """Ctrl+C 처리"""
    print("\n🛑 학습이 중단되었습니다. 체크포인트는 저장되었습니다.")
    sys.exit(0)

def check_system_requirements():
    """시스템 요구사항 확인"""
    logger = logging.getLogger(__name__)
    
    # Python 버전 확인
    python_version = sys.version_info
    logger.info(f"🐍 Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # PyTorch 확인
    logger.info(f"🔥 PyTorch 버전: {torch.__version__}")
    
    # CUDA 확인
    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"   CUDA 버전: {torch.version.cuda}")
    else:
        logger.warning("⚠️  CUDA를 사용할 수 없습니다. CPU로 학습합니다.")

def validate_config(config):
    """설정 파일 유효성 검증"""
    logger = logging.getLogger(__name__)
    
    required_keys = {
        'data': ['path', 'batch_size'],
        'model': ['type'],
        'training': ['epochs', 'learning_rate'],
        'output_path': None
    }
    
    missing_keys = []
    
    for key, subkeys in required_keys.items():
        if key not in config:
            missing_keys.append(key)
        elif subkeys:
            for subkey in subkeys:
                if subkey not in config[key]:
                    missing_keys.append(f"{key}.{subkey}")
    
    if missing_keys:
        logger.error(f"❌ 설정 파일에 누락된 키들: {missing_keys}")
        return False
    
    # 데이터 경로 확인
    data_path = config['data']['path']
    if not os.path.exists(data_path):
        logger.error(f"❌ 데이터 경로를 찾을 수 없습니다: {data_path}")
        return False
    
    logger.info("✅ 설정 파일 검증 완료")
    return True

def create_output_directories(config):
    """필요한 출력 디렉토리들 생성"""
    logger = logging.getLogger(__name__)
    
    directories = [
        config['output_path'],
        config['training']['checkpoint_dir'],
        config['training']['log_dir']
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 디렉토리 생성: {directory}")

def print_training_info(config):
    """학습 정보 출력"""
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 === SSD-NeRF 학습 정보 ===")
    logger.info(f"   모델 타입: {config['model']['type']}")
    logger.info(f"   에포크 수: {config['training']['epochs']}")
    logger.info(f"   배치 크기: {config['data']['batch_size']}")
    logger.info(f"   학습률: {config['training']['learning_rate']}")
    logger.info(f"   데이터 경로: {config['data']['path']}")
    logger.info(f"   출력 경로: {config['output_path']}")
    logger.info("="*50)

def main():
    # 로깅 설정
    logger = setup_logging()
    
    # Ctrl+C 처리
    signal.signal(signal.SIGINT, signal_handler)
    
    # 시작 시간 기록
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    parser = argparse.ArgumentParser(
        description="🚀 SSD-NeRF 학습 스크립트 (개선된 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 사용법:
  python scripts/run_training.py --config configs/default_config.py
  python scripts/run_training.py --config configs/kitti_config.yaml
  python scripts/run_training.py --config configs/static_config.py --resume
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help="설정 파일 경로 (.py 또는 .yaml)")
    parser.add_argument('--resume', action='store_true',
                        help="체크포인트에서 학습 재시작")
    parser.add_argument('--gpu', type=int, default=None,
                        help="사용할 GPU ID (기본값: 자동 선택)")
    
    args = parser.parse_args()
    
    try:
        logger.info(f"🚀 SSD-NeRF 학습 시작 - {start_datetime}")
        
        # 1. 시스템 요구사항 확인
        logger.info("🔍 시스템 요구사항 확인 중...")
        check_system_requirements()
        
        # 2. 설정 파일 로드
        logger.info(f"📋 설정 파일 로드 중: {args.config}")
        if not os.path.exists(args.config):
            logger.error(f"❌ 설정 파일을 찾을 수 없습니다: {args.config}")
            sys.exit(1)
        
        config = load_config(args.config)
        
        # 3. 설정 파일 검증
        logger.info("✅ 설정 파일 검증 중...")
        if not validate_config(config):
            sys.exit(1)
        
        # 4. GPU 설정
        if args.gpu is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            logger.info(f"🎯 GPU {args.gpu} 사용 설정")
        
        # 5. 출력 디렉토리 생성
        logger.info("📁 출력 디렉토리 생성 중...")
        create_output_directories(config)
        
        # 6. 학습 정보 출력
        print_training_info(config)
        
        # 7. 트레이너 초기화
        logger.info("🔧 트레이너 초기화 중...")
        trainer = Trainer(config)
        
        # 8. 재시작 옵션 처리
        if args.resume:
            logger.info("🔄 체크포인트에서 학습 재시작...")
        
        # 9. 학습 시작
        logger.info("🚀 학습 시작!")
        trainer.train()
        
        # 10. 학습 완료
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"🎉 학습 완료! 소요 시간: {duration/3600:.2f}시간")
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 학습이 중단되었습니다.")
        sys.exit(0)
        
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없습니다: {e}")
        sys.exit(1)
        
    except ImportError as e:
        logger.error(f"❌ 모듈 임포트 오류: {e}")
        logger.error("필요한 패키지들이 설치되어 있는지 확인하세요.")
        sys.exit(1)
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            logger.error(f"❌ CUDA 오류: {e}")
            logger.error("GPU 메모리가 부족하거나 CUDA 설정에 문제가 있습니다.")
        else:
            logger.error(f"❌ 런타임 오류: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        logger.error("자세한 오류 정보:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 