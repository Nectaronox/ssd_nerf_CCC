import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

from .ssd_model import SSDModel

class PredictionHead(nn.Module):
    """
    A simple MLP to predict 3D box parameters from RoI-aligned features.
    """
    def __init__(self, in_channels, num_params):
        super(PredictionHead, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_params)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SSDNeRF(nn.Module):
    """
    An object-centric 3D object detection model.
    It uses an SSD for 2D proposals and a head to predict 3D box parameters
    from RoI-aligned features.
    """
    def __init__(self, num_classes, box_params=7, input_size=300):
        super(SSDNeRF, self).__init__()
        self.ssd = SSDModel(num_classes=num_classes)
        self.input_size = input_size
        
        # Calculate spatial_scale dynamically based on feature map reduction
        # VGG conv4_3 has stride 8, so spatial_scale = 1/8
        # But we make it configurable for different input sizes
        self.spatial_scale = 1.0 / 8.0  # Default for VGG conv4_3
        
        # RoIAlign to extract features for each box from the first VGG feature map
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=self.spatial_scale, sampling_ratio=2)
        
        # The prediction head
        # Input channels: 512 (from VGG feature map) * 7 * 7 (from RoIAlign)
        self.head = PredictionHead(512 * 7 * 7, box_params)

    def forward(self, image, proposals_list=None):
        """
        Args:
            image (Tensor): A batch of images (N, C, H, W).
            proposals_list (list of Tensors), optional:
                A list of 2D proposal boxes for each image in the batch.
                Each tensor is of shape [num_boxes, 4] in (x1, y1, x2, y2) format.
                If provided (i.e., during training), the model will predict 3D boxes for these proposals.
                If None (i.e., during inference), the model only returns the 2D detections.

        Returns:
            A tuple containing:
            - locs_2d (Tensor): 2D location predictions from SSD.
            - confs_2d (Tensor): 2D class predictions from SSD.
            - pred_3d_params (Tensor or None): Predicted 3D box parameters for each proposal.
        """
        locs_2d, confs_2d, ssd_features = self.ssd(image)
        
        # We use the first feature map from the SSD backbone (VGG-16 conv4_3)
        feature_map = ssd_features[0]

        if proposals_list is None:
            # Inference path
            return locs_2d, confs_2d, None

        # Training path: use provided ground-truth 2D boxes as proposals.
        # Create batch indices for RoIAlign - 수정된 부분
        box_indices = []
        for i, proposals in enumerate(proposals_list):
            if len(proposals) > 0:
                batch_indices = torch.full((proposals.shape[0], 1), i, 
                                         dtype=torch.float, device=image.device)
                box_indices.append(batch_indices)
        
        if not box_indices:
            # No proposals in any image
            return locs_2d, confs_2d, None
            
        box_indices = torch.cat(box_indices, dim=0)
        
        # Concatenate all proposals into a single tensor
        all_proposals = torch.cat(proposals_list, dim=0)
        
        # Combine indices and proposals for RoIAlign
        rois = torch.cat([box_indices, all_proposals], dim=1)

        # Extract features for each proposal
        object_features = self.roi_align(feature_map, rois)

        # Predict 3D box parameters
        pred_3d_params = self.head(object_features)

        return locs_2d, confs_2d, pred_3d_params