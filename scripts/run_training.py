import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import Trainer
from src.utils.config_utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Run SSD-NeRF Training")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the configuration file (e.g., configs/default_config.py)")
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 