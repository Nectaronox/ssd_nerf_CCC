
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models import vgg16

class SSDModel(nn.Module):
    """
    A simplified SSD (Single Shot Detector) model for 2D object detection.
    This implementation uses a VGG16 backbone.
    """
    def __init__(self, num_classes):
        super(SSDModel, self).__init__()
        self.num_classes = num_classes
        
        # Use a pretrained VGG16 model as the backbone
        self.backbone = vgg16(pretrained=True).features[:-1]
        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Additional feature layers for SSD
        self.extras = self._add_extras()
        
    
        # Loc and Conf layers for each feature map
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        
        # Define anchor boxes per location for each feature map
        # This is a simplified configuration
        mbox = [4, 6, 6, 6, 4, 4] 
        
        # Feature maps from VGG and extras
        vgg_source = [21, -1]
        
        # Add loc and conf layers
        for i, feature_layer in enumerate(vgg_source):
            self.loc_layers.append(nn.Conv2d(self.backbone[feature_layer].out_channels,
                                             mbox[i] * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.backbone[feature_layer].out_channels,
                                              mbox[i] * self.num_classes, kernel_size=3, padding=1))

        for i, feature_layer in enumerate(self.extras, 2):
            if isinstance(feature_layer, nn.Conv2d):
                self.loc_layers.append(nn.Conv2d(feature_layer.out_channels, mbox[i] * 4, kernel_size=3, padding=1))
                self.conf_layers.append(nn.Conv2d(feature_layer.out_channels, mbox[i] * self.num_classes, kernel_size=3, padding=1))

    def _add_extras(self):
        layers = []
        in_channels = 512 # VGG16's last feature map
        
        layers += [nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(1024, 1024, kernel_size=1)]
        layers += [nn.ReLU(inplace=True)]
        
        # Add more layers for multi-scale detection
        layers += [nn.Conv2d(1024, 256, kernel_size=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]
        layers += [nn.ReLU(inplace=True)]

        return nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input image tensor of shape (N, C, H, W).
        
        Returns:
            tuple: A tuple containing:
                - locs (list of Tensors): Location predictions for each feature map.
                - confs (list of Tensors): Classification predictions for each feature map.
                - features (list of Tensors): The feature maps themselves.
        """
        features = []
        locs = []
        confs = []

        # Pass through VGG backbone
        for i in range(23):
            x = self.backbone[i](x)
        features.append(x)

        # Pass through remaining VGG layers
        for i in range(23, len(self.backbone)):
            x = self.backbone[i](x)
        features.append(x)
        
        # Pass through extra layers
        for i, l in enumerate(self.extras):
            x = F.relu(l(x), inplace=True)
            if i % 2 == 1: # Add feature map after every other conv layer
                features.append(x)

        # Apply loc and conf layers to each feature map
        for (x, l, c) in zip(features, self.loc_layers, self.conf_layers):
            locs.append(l(x).permute(0, 2, 3, 1).contiguous())
            confs.append(c(x).permute(0, 2, 3, 1).contiguous())
            
        # Reshape to (N, num_boxes, 4) and (N, num_boxes, num_classes)
        locs = torch.cat([o.view(o.size(0), -1) for o in locs], 1)
        confs = torch.cat([o.view(o.size(0), -1) for o in confs], 1)

        locs = locs.view(locs.size(0), -1, 4)
        confs = confs.view(confs.size(0), -1, self.num_classes)
        
        return locs, confs, features
