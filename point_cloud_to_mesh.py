import numpy as np
import torch
import os
import open3d as o3d
import trimesh
from plyfile import PlyData
import matplotlib.pyplot as plt

class PointCloudToMeshConverter:
    def __init__(self, point_clouds_directory):
        self.point_clouds_directory = point_clouds_directory
        self.point_clouds = []
        self.meshes = []
        self.load_point_clouds_from_ply()
        self.create_meshes_without_normal_orientation()

    def load_point_clouds_from_ply(self):
        point_cloud_files = [f for f in os.listdir(self.point_clouds_directory) if f.endswith('.ply')]
        for f in point_cloud_files:
            file_path = os.path.join(self.point_clouds_directory, f)
            ply_data = PlyData.read(file_path)
            x = ply_data['vertex']['x']
            y = ply_data['vertex']['y']
            z = ply_data['vertex']['z']
            points = torch.tensor(np.column_stack((x, y, z)), dtype=torch.float32)
            self.point_clouds.append(points)
    
    @staticmethod
    def calculate_density_percentile(num_points):
        max_threshold = 50
        min_threshold = 20
        max_points = 9000
        min_points = 2600
        num_points = np.clip(num_points, min_points, max_points)
        threshold = max_threshold - (num_points - min_points) * (max_threshold - min_threshold) / (max_points - min_points)
        return threshold
    
    def create_meshes_without_normal_orientation(self):
        for point_cloud_tensor in self.point_clouds:
            point_cloud_np = point_cloud_tensor.cpu().numpy()
            density_percentile = self.calculate_density_percentile(len(point_cloud_np))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            densities = np.asarray(densities)
            density_threshold = np.percentile(densities, density_percentile)
            mesh = mesh.select_by_index((densities > density_threshold).nonzero()[0])
            mesh.paint_uniform_color([1, 0.706, 0])
            self.meshes.append(mesh)

    @staticmethod
    def open3d_mesh_to_trimesh(open3d_mesh):
        vertices = np.asarray(open3d_mesh.vertices)
        faces = np.asarray(open3d_mesh.triangles)
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return trimesh_mesh
    
    def get_trimeshes(self):
        trimesh_list = [self.open3d_mesh_to_trimesh(mesh) for mesh in self.meshes]
        return trimesh_list
    
    def get_open3Dmeshes(self):
        return self.meshes
    
    def visualize_meshes(self):
        num_meshes = len(self.meshes)
        num_cols = 3
        num_rows = (num_meshes + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        for i, (ax, mesh) in enumerate(zip(axes, self.meshes)):
            ax.set_title(f"Mesh {i+1}")
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            if vertices.shape[0] == 0 or triangles.shape[0] == 0:
                print(f"Mesh {i+1} has zero-size arrays.")
                continue
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, cmap='viridis')
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    print("Hello 1")
    # Initialize the converter with the directory containing your point clouds
    converter = PointCloudToMeshConverter('/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/data/B73-selected')
    print("Hello 2")
    # Get the list of trimesh objects
    trimeshes = converter.get_trimeshes()
    print("Hello 3")
    # Now you can work with the list of trimeshes for further processing or analysis
    # For example, you could save each trimesh to a file
    for i, trimesh in enumerate(trimeshes):
        # trimesh.export(f'mesh_{i}.ply')
        print(f'{i}')

    # converter.visualize_meshes()
