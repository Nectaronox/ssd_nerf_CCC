# SSD-NeRF Default Configuration for KITTI

config = {
    'data': {
        'path': 'data/kitti',  # Path to the KITTI dataset directory
        'batch_size': 1, # Process one sequence at a time
        'num_workers': 4,
        'image_size': [375, 1242], # KITTI image dimensions (can be downsampled)
        'lidar_points': 8192,   # Number of LiDAR points to sample
        'focal_length': 721.5377,  # KITTI 카메라 초점 거리 (P2 기본값)
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
        'checkpoint_interval': 10,  # 체크포인트 저장 주기 (epoch 단위)
        'log_dir': 'output/logs',
        
        # ✅ rays.py 고급 기능 설정
        'num_train_rays': 1024,    # 훈련에 사용할 ray 수 (메모리와 성능의 균형)
        'use_disparity_sampling': True,  # disparity space에서 샘플링 (근거리 물체에 유리)
        'use_ndc': False,          # NDC 변환 사용 (무한 거리 씬에 유용, KITTI는 일반적으로 False)
        
        # 추가 최적화 설정
        'gradient_clip_max_norm': 1.0,  # 그래디언트 클리핑
        'loss_weights': {
            'nerf': 1.0,           # NeRF 렌더링 loss 가중치
            'diffusion': 0.1,      # Diffusion loss 가중치  
            'displacement': 0.01   # Displacement regularization 가중치
        }
    },
    'output_path': 'output',  # 누락된 출력 경로 추가
    
    # ✅ rays.py 기능 활용을 위한 상세 설정
    'rendering': {
        'perturb_rays': True,      # 훈련 중 ray 샘플링에 noise 추가
        'white_background': False, # KITTI는 자연 배경이므로 False
        'raw_noise_std': 0.1,     # density prediction에 추가할 noise
    }
} 