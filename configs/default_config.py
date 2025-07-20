# SSD-NeRF Default Configuration for KITTI

config = {
    'data': {
        'path': 'data/kitti',  # Path to the KITTI dataset directory
        'batch_size': 1, # Process one sequence at a time
        'num_workers': 4,
        'image_size': [375, 1242], # KITTI image dimensions (can be downsampled)
        'lidar_points': 8192,   # Number of LiDAR points to sample
    },
    'renderer': {
        'n_samples': 128,
        'near': 0.5,
        'far': 200.0,
    },
    'model': {
        'type': 'dynamic',  # 'dynamic' or 'static' - choose SSD-NeRF variant
        'diffusion': {
            'time_steps': 1000,
            'feature_dim': 128,
        },
        'nerf': {
            'embedding_dim': 256,
            'num_layers': 8,
            'use_viewdirs': True,
        },
        'ssd_nerf': {
            'input_dim': 3, # x, y, z for LiDAR
            'output_dim': 4, # r, g, b, density
            'num_classes': 3, # Car, Van, Truck for KITTI
        }
    },
    'training': {
        'learning_rate': 1e-4,
        'epochs': 100,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_step_size': 30,
        'scheduler_gamma': 0.1,
        'checkpoint_dir': 'output/checkpoints',
        'log_dir': 'output/logs',
    },
    'output_path': 'output'  # 누락된 출력 경로 추가
} 