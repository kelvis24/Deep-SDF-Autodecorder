

from PointCloudProcessor import PointCloudProcessor
from custom_mesh_to_sdf import MeshToSDFConverter
from dataloader import ShapeNet_Dataset
from model import Decoder
from point_cloud_to_mesh import PointCloudToMeshConverter
from reconstruct import reconstruct
from train import train_decoder
import torch
import matplotlib.pyplot as plt
import os

def plot_point_cloud(mesh, title="Point Cloud", ax=None):
    points = mesh.vertices  # Extract vertices from the mesh

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)  # 's' is the size of each point
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    
    # pc2m = PointCloudToMeshConverter('/Users/elviskimara/Downloads/DeepSDFcode/data/B73-selected')

    # # Get the list of trimesh objects
    # meshes = pc2m.get_open3Dmeshes()
    
    

    # # # Example usage
    # converter = MeshToSDFConverter(target_directory="./processed_data/train/")
    # converter.process_meshes(meshes)

    # for i, point_cloud in enumerate(meshes):
    #     plot_point_cloud(pc2m.open3d_mesh_to_trimesh(point_cloud), title=f"Point Cloud {i + 1}")
        
        


    # checkpoint_save_path = "/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/checkpoints"

    # # Check if the directory exists
    # if not os.path.exists(checkpoint_save_path):
    #     # If not, create it
    #     os.makedirs(checkpoint_save_path)


    # train_decoder()

    # # ------------ setting device on GPU if available, else CPU ------------
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)

    # # ------------ load validation samples ------------
    # val_data_path = "/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/processed_data/train"
    # val_dataset = ShapeNet_Dataset(val_data_path)

    # # ------------ load decoder ------------
    # decoder = Decoder().to(device)
    # checkpoint = torch.load("/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/checkpoints/checkpoints100.pt")
    # decoder.load_state_dict(checkpoint["model"])


    # reconstruction_path = "/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/reconstruction/"

    # # Check if the directory exists
    # if not os.path.exists(reconstruction_path):
    #     # If not, create it
    #     os.makedirs(reconstruction_path)

    # # ------------ reconstruction ------------
    # for idx in range(len(val_dataset)):

    #     test_sample = val_dataset[idx]

    #     filename = "/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/reconstruction/reconstruction" + str(12+idx)
    #     reconstruct(test_sample,
    #                             decoder,
    #                             filename,
    #                             lat_iteration=1000,
    #                             lat_init_std = 0.01,
    #                             lat_lr = 5e-4,
    #                             N=256,
    #                             max_batch=32 ** 3)
        
    #     # Example usage:
    point_clouds_directory = '/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/reconstruction'
    processor = PointCloudProcessor(point_clouds_directory)

    for i, point_cloud in enumerate(processor.point_clouds):
        print("1")
        processor.plot_point_cloud_as_image(point_cloud, title=f"Point Cloud {i + 1}", file_name=f"reconstruction_image_{i+1}")