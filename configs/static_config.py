# Static SSD-NeRF Configuration for comparison

config = {
    'data': {
        'path': 'data/kitti',
        'batch_size': 1,
        'num_workers': 4,
        'image_size': [375, 1242],
        'lidar_points': 8192,
    },
    'renderer': {
        'n_samples': 128,
        'near': 0.5,
        'far': 200.0,
    },
    'model': {
        'type': 'static',  # Use traditional static SSD-NeRF
        'ssd_nerf': {
            'input_dim': 3,
            'output_dim': 4,
            'num_classes': 3,  # Car, Van, Truck
        }
    },
    'training': {
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'scheduler_step_size': 30,
        'scheduler_gamma': 0.1,
        'checkpoint_dir': 'output/checkpoints_static',
        'log_dir': 'output/logs_static',
    },
    'output_path': 'output'  # 누락된 출력 경로 추가
} 