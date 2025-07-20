"""
Comprehensive SSD-NeRF Model Evaluation Script
Evaluates and compares Dynamic vs Static SSD-NeRF models
Enhanced with better error handling and logging
"""

import argparse
import sys
import os
import logging
import traceback

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import SSDNeRFBenchmark
from src.data.dataset import KITTIDataset
from src.utils.config_utils import load_config

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_inputs(args):
    """입력 파라미터 검증"""
    logger = logging.getLogger(__name__)
    
    # Config 파일 확인
    if not os.path.exists(args.config):
        logger.error(f"❌ Config file not found: {args.config}")
        return False
    
    # Checkpoint 파일 확인
    if args.model_type in ['dynamic', 'both'] and args.dynamic_checkpoint:
        if not os.path.exists(args.dynamic_checkpoint):
            logger.error(f"❌ Dynamic checkpoint not found: {args.dynamic_checkpoint}")
            return False
    
    if args.model_type in ['static', 'both'] and args.static_checkpoint:
        if not os.path.exists(args.static_checkpoint):
            logger.error(f"❌ Static checkpoint not found: {args.static_checkpoint}")
            return False
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Output directory ready: {output_dir}")
    
    return True

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description="🔍 Comprehensive SSD-NeRF Model Evaluation (Enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시 사용법:
  # Dynamic 모델만 평가
  python scripts/run_evaluation.py --config configs/default_config.py --dynamic_checkpoint output/checkpoints/dynamic_model.pth --model_type dynamic
  
  # Static 모델만 평가  
  python scripts/run_evaluation.py --config configs/default_config.py --static_checkpoint output/checkpoints/static_model.pth --model_type static
  
  # 두 모델 비교
  python scripts/run_evaluation.py --config configs/default_config.py --dynamic_checkpoint dynamic.pth --static_checkpoint static.pth --model_type both
        """
    )
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--dynamic_checkpoint', type=str, help="Path to dynamic model checkpoint")
    parser.add_argument('--static_checkpoint', type=str, help="Path to static model checkpoint")
    parser.add_argument('--model_type', type=str, choices=['dynamic', 'static', 'both'], default='both',
                        help="Which model(s) to evaluate")
    parser.add_argument('--max_samples', type=int, default=100, help="Maximum samples to evaluate")
    parser.add_argument('--output', type=str, default='output/evaluation_results.json', 
                        help="Output file for results")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help="Dataset split to evaluate on")
    
    args = parser.parse_args()
    
    try:
        # 입력 검증
        logger.info("🔍 Validating inputs...")
        if not validate_inputs(args):
            return 1
        
        # Load configuration
        logger.info(f"📋 Loading configuration from: {args.config}")
        config = load_config(args.config)
        logger.info("✅ Configuration loaded successfully")
        
        # Load dataset
        logger.info(f"📊 Loading KITTI dataset ({args.split} split)")
        try:
            dataset = KITTIDataset(config, split=args.split)
            logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
            
            if len(dataset) == 0:
                logger.warning("⚠️ Dataset is empty. Please check data path and split.")
                return 1
                
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        # Initialize benchmark
        logger.info("🔧 Initializing evaluation benchmark...")
        try:
            benchmark = SSDNeRFBenchmark(config, dataset)
            logger.info("✅ Benchmark initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize benchmark: {e}")
            logger.error(traceback.format_exc())
            return 1
        
        # ✅ 개선된 평가 실행 로직
        results = None
        
        if args.model_type == 'both':
            # 두 모델 비교
            if not args.dynamic_checkpoint or not args.static_checkpoint:
                logger.error("❌ Both dynamic and static checkpoint paths required for comparison")
                return 1
            
            logger.info("🥊 Running comprehensive comparison between Dynamic and Static models")
            try:
                results = benchmark.compare_models(
                    args.dynamic_checkpoint, 
                    args.static_checkpoint, 
                    args.max_samples
                )
            except Exception as e:
                logger.error(f"❌ Model comparison failed: {e}")
                logger.error(traceback.format_exc())
                return 1
                
        elif args.model_type == 'dynamic':
            # Dynamic 모델만 평가
            if not args.dynamic_checkpoint:
                logger.error("❌ Dynamic checkpoint path required")
                return 1
            
            logger.info("🚀 Evaluating Dynamic SSD-NeRF model")
            try:
                results = benchmark.evaluate_model(args.dynamic_checkpoint, 'dynamic', args.max_samples)
            except Exception as e:
                logger.error(f"❌ Dynamic model evaluation failed: {e}")
                logger.error(traceback.format_exc())
                return 1
                
        elif args.model_type == 'static':
            # Static 모델만 평가
            if not args.static_checkpoint:
                logger.error("❌ Static checkpoint path required")
                return 1
            
            logger.info("📷 Evaluating Static SSD-NeRF model")
            try:
                results = benchmark.evaluate_model(args.static_checkpoint, 'static', args.max_samples)
            except Exception as e:
                logger.error(f"❌ Static model evaluation failed: {e}")
                logger.error(traceback.format_exc())
                return 1
        
        # Save results
        if results:
            try:
                benchmark.save_results(args.output)
                logger.info(f"💾 Results saved to: {args.output}")
            except Exception as e:
                logger.error(f"❌ Failed to save results: {e}")
                logger.error(traceback.format_exc())
                return 1
        
        logger.info("🎉 Evaluation complete!")
        
        # Print quick summary
        if 'comparison' in benchmark.results and benchmark.results['comparison']:
            comp = benchmark.results['comparison']
            recommendation = comp.get('recommendation', {})
            if recommendation:
                logger.info(f"🏆 FINAL RECOMMENDATION: {recommendation.get('recommended_model', 'N/A').upper()}")
                logger.info(f"💡 {recommendation.get('reasoning', 'No reasoning provided')}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 