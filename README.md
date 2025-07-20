# SSD-NeRF: Single-Stage-Diffusion NeRF for Autonomous Driving

This repository contains the official implementation of SSD-NeRF, a novel method that combines a diffusion model and a Neural Radiance Field (NeRF) in a single stage for robust 3D scene reconstruction, particularly for assisting LiDAR in autonomous driving scenarios.

## Features

- **End-to-End Training**: Single-stage training of diffusion and NeRF models
- **High-Quality 3D Reconstruction**: Generates high-fidelity neural radiance fields from sparse inputs
- **LiDAR Augmentation**: Designed to enhance and densify sparse LiDAR point clouds
- **KITTI Dataset Support**: Optimized for KITTI autonomous driving dataset

## Requirements

- Python 3.12+ (recommended) or Python 3.9+
- PyTorch >= 1.9.0
- CUDA compatible GPU (optional but recommended)

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd SSD_NeRF_Computer
```

2. **Create virtual environment (Python 3.12 recommended):**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS  
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Setup

### KITTI Dataset

This implementation uses the KITTI 3D Object Detection dataset.

#### Option 1: Generate Sample Data (Quick Test)
```bash
python scripts/download_kitti_data.py --sample
```

#### Option 2: Download Real KITTI Data
```bash
# Download subset for testing (recommended)
python scripts/download_kitti_data.py

# Download full dataset (~50GB)
python scripts/download_kitti_data.py --all
```

#### Option 3: Manual Download
1. Visit [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
2. Register and download:
   - Left color images of object data set (12 GB)
   - Velodyne point clouds (29 GB)  
   - Camera calibration matrices (16 MB)
3. Extract to `data/kitti/` following this structure:
```
data/kitti/
├── object/
│   ├── training/
│   │   ├── image_2/     # Left color camera images
│   │   ├── velodyne/    # LiDAR point clouds (.bin files)
│   │   └── calib/       # Camera calibration files
│   └── testing/
│       ├── image_2/
│       ├── velodyne/
│       └── calib/
```

### Dataset Verification
```bash
python src/data/dataset.py
```

## Usage

### Training

To train a new model:

```bash
python scripts/run_training.py --config configs/default_config.py
```

### Inference

To perform inference with a trained model:

```bash
python scripts/run_inference.py --checkpoint_path output/model.pth
```

## Configuration

Key configuration options in `configs/default_config.py`:

- `data.path`: Path to KITTI dataset (`data/kitti`)
- `data.image_size`: Input image dimensions `[375, 1242]` (KITTI standard)
- `data.lidar_points`: Number of LiDAR points to sample (`8192`)
- `training.learning_rate`: Learning rate (`1e-4`)
- `training.num_epochs`: Number of training epochs (`100`)

## Project Structure

```
SSD_NeRF_Computer/
├── configs/                 # Configuration files
├── data/                   # Dataset storage
├── src/
│   ├── data/              # Dataset loaders
│   ├── models/            # Model implementations
│   ├── training/          # Training utilities
│   └── utils/             # Utility functions
├── scripts/               # Training and inference scripts
├── output/                # Model outputs and logs
└── requirements.txt       # Dependencies
```

## Troubleshooting

### Python Version Issues
- This implementation requires Python 3.9+ (3.12 recommended)
- If using Python 3.13, some dependencies may not be available

### Memory Issues
- Reduce `data.lidar_points` in config if running out of memory
- Use smaller `data.batch_size`

### Dataset Issues
- Ensure KITTI data follows the correct directory structure
- Run dataset verification script to check data integrity

## License

[License information here]

## Citation

[Citation information here] 