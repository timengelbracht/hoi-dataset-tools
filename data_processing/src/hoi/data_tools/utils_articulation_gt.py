import numpy as np
import open3d as o3d

def get_pcd_diff(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud, threshold: float) -> o3d.geometry.PointCloud:
    """
    Computes the difference between two point clouds.
    Points in pcd1 that are farther than threshold from any point in pcd2 are kept.
    """
    distances = pcd1.compute_point_cloud_distance(pcd2)
    mask = np.array(distances) > threshold
    diff_pcd = pcd1.select_by_index(np.where(mask)[0])
    return diff_pcd

def visualize_pcds(pcds: list[o3d.geometry.PointCloud], colors: list[np.ndarray]):
    """
    Visualizes multiple point clouds with specified colors.
    """
    for pcd, color in zip(pcds, colors):
        pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries(pcds)
    