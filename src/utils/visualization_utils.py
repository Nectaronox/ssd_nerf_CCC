import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_point_cloud(points, colors=None):
    """
    Visualizes a point cloud using Open3D.
    
    Args:
        points (np.ndarray or torch.Tensor): A (N, 3) array of point coordinates.
        colors (np.ndarray or torch.Tensor, optional): A (N, 3) array of RGB colors for each point. 
                                                        Values should be in range [0, 1].
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

def plot_rendered_image(image_tensor):
    """
    Plots a rendered image from a NeRF model.
    
    Args:
        image_tensor (torch.Tensor): A (H, W, 3) tensor representing the image.
    """
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu().numpy()
    
    plt.imshow(image_tensor)
    plt.title("Rendered Image")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # --- Example for Point Cloud Visualization ---
    print("Displaying a random point cloud...")
    # Create a random point cloud
    random_points = np.random.rand(1000, 3) * 10
    # Create random colors
    random_colors = np.random.rand(1000, 3)
    
    visualize_point_cloud(random_points, colors=random_colors)
    
    # --- Example for Image Plotting ---
    print("Displaying a random image tensor...")
    # Create a random image tensor (H, W, C)
    random_image = torch.rand(128, 256, 3)
    
    plot_rendered_image(random_image) 