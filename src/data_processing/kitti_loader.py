import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class KittiDataset(Dataset):
    """
    KITTI dataset loader for 3D object detection.
    """
    def __init__(self, base_dir: str, scene: str, transform=None):
        """
        Args:
            base_dir (str): The base directory of the KITTI dataset.
            scene (str): The scene (e.g., '00', '01', ...) to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.base_dir = os.path.join(base_dir, 'sequences', scene)
        self.image_dir = os.path.join(self.base_dir, 'image_2')
        self.label_dir = os.path.join(self.base_dir, 'label_2')
        self.pose_path = os.path.join(base_dir, 'poses', f'{scene}.txt')
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
        self.poses = self._load_poses()
        self.focal = self._load_focal()

        if len(self.image_files) != len(self.poses) or len(self.image_files) != len(self.label_files):
            raise ValueError("Number of images, poses, and labels do not match.")

    def _load_poses(self) -> np.ndarray:
        """Loads camera poses."""
        poses = []
        with open(self.pose_path, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float32, sep=' ')
                T = T.reshape(3, 4)
                poses.append(T)
        return np.array(poses)

    def _load_focal(self) -> float:
        """Loads focal length from calibration file."""
        calib_path = os.path.join(self.base_dir, 'calib.txt')
        with open(calib_path, 'r') as f:
            line = f.readlines()[0] # P2 for the left color camera
            P2 = np.fromstring(line.split(': ')[1], dtype=np.float32, sep=' ')
            focal = P2[0]
        return focal

    def _load_labels(self, idx: int) -> list:
        """Loads 3D bounding box labels for a given index."""
        labels = []
        with open(os.path.join(self.label_dir, self.label_files[idx]), 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(' ')
                if parts[0] in ['Car', 'Van', 'Truck']: # Focus on vehicles
                    labels.append({
                        'class': parts[0],
                        'truncation': float(parts[1]),
                        'occlusion': int(parts[2]),
                        'alpha': float(parts[3]),
                        'bbox_2d': [float(p) for p in parts[4:8]],
                        'dimensions': [float(p) for p in parts[8:11]], # h, w, l
                        'location': [float(p) for p in parts[11:14]], # x, y, z
                        'rotation_y': float(parts[14])
                    })
        return labels

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a sample from the dataset.
        """
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        pose = torch.from_numpy(self.poses[idx])
        labels = self._load_labels(idx)

        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'pose': pose,
            'focal': self.focal,
            'labels': labels,
            'bbox_2d': [l['bbox_2d'] for l in labels]
        }
        
        return sample 