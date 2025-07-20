"""
Comprehensive SSD-NeRF Model Evaluation Script
Evaluates and compares Dynamic vs Static SSD-NeRF models
"""

import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import SSDNeRFBenchmark
from src.data.dataset import KITTIDataset
from src.utils.config_utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSD-NeRF Models")
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
    
    # Load configuration
    print(f"📋 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Load dataset
    print(f"📊 Loading KITTI dataset ({args.split} split)")
    try:
        dataset = KITTIDataset(config, split=args.split)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Initialize benchmark
    benchmark = SSDNeRFBenchmark(config, dataset)
    
    # Run evaluation based on arguments
    if args.model_type == 'both' and args.dynamic_checkpoint and args.static_checkpoint:
        print("🥊 Running comprehensive comparison between Dynamic and Static models")
        results = benchmark.compare_models(
            args.dynamic_checkpoint, 
            args.static_checkpoint, 
            args.max_samples
        )
        
    elif args.model_type == 'dynamic' or (args.model_type == 'both' and args.dynamic_checkpoint):
        if not args.dynamic_checkpoint:
            print("❌ Dynamic checkpoint path required")
            return
        print("🚀 Evaluating Dynamic SSD-NeRF model")
        results = benchmark.evaluate_model(args.dynamic_checkpoint, 'dynamic', args.max_samples)
        
    elif args.model_type == 'static' or (args.model_type == 'both' and args.static_checkpoint):
        if not args.static_checkpoint:
            print("❌ Static checkpoint path required")
            return
        print("📷 Evaluating Static SSD-NeRF model")
        results = benchmark.evaluate_model(args.static_checkpoint, 'static', args.max_samples)
        
    else:
        print("❌ Please provide appropriate checkpoint path(s)")
        return
    
    # Save results
    benchmark.save_results(args.output)
    
    print(f"\n🎉 Evaluation complete! Results saved to: {args.output}")
    
    # Print quick summary
    if 'comparison' in benchmark.results and benchmark.results['comparison']:
        comp = benchmark.results['comparison']
        print(f"\n🏆 FINAL RECOMMENDATION: {comp['recommendation']['recommended_model'].upper()}")
        print(f"💡 {comp['recommendation']['reasoning']}")

if __name__ == '__main__':
    main() 