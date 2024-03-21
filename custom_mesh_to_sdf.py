import os
import numpy as np
import open3d as o3d
from point_cloud_to_mesh import PointCloudToMeshConverter
from mesh_to_sdf import sample_sdf_near_surface
os.environ['PYOPENGL_PLATFORM'] = 'egl'

class MeshToSDFConverter:
    def __init__(self, target_directory="./processed_data/train/"):
        self.target_directory = target_directory
        # Set environment variable for PyOpenGL platform to 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensures the target directory exists, creates it if it does not."""
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

    def generate_xyz_sdf(self, mesh):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        """Generates XYZ coordinates and SDF values for a given mesh."""
        # Compute the bounding box and its centroid
        bbox = mesh.get_axis_aligned_bounding_box()
        centroid = np.asarray(bbox.get_center())

        # Adjust the mesh vertices based on the centroid
        vertices = np.asarray(mesh.vertices)
        distances = np.linalg.norm(vertices - centroid, axis=1)
        max_distance = np.max(distances)
        mesh.vertices = o3d.utility.Vector3dVector((vertices - centroid) / max_distance)

        # Proceed with sampling
        xyz, sdf = sample_sdf_near_surface(pc2m.open3d_mesh_to_trimesh(mesh), number_of_points=15000)
        return xyz, sdf

    @staticmethod
    def write_sdf_to_npz(xyz, sdfs, filename):
        """Writes the SDF data to an NPZ file."""
        num_vert = len(xyz)
        pos, neg = [], []

        for i in range(num_vert):
            v = xyz[i]
            s = sdfs[i]

            if s > 0:
                pos.extend(list(v) + [s])
            else:
                neg.extend(list(v) + [s])

        np.savez(filename, pos=np.array(pos).reshape(-1, 4), neg=np.array(neg).reshape(-1, 4))

    def process_mesh(self, mesh, mesh_name):
        """Processes a single mesh and writes the output to an NPZ file."""
        target_filepath = os.path.join(self.target_directory, f"{mesh_name}_processed.npz")
        xyz, sdfs = self.generate_xyz_sdf(mesh)
        self.write_sdf_to_npz(xyz, sdfs, target_filepath)
        print("Process finished:", mesh_name)

    def process_meshes(self, meshes):
        """Processes a list of meshes."""
        N = len(meshes)
        print(N)

        for it, mesh in enumerate(meshes):
            mesh_name = f"mesh{it + 1}"
            self.process_mesh(mesh, mesh_name)
            print("Processed:", mesh_name, it + 1, "/", N)


if __name__ == "__main__":
    # Initialize the converter with the directory containing your point clouds
    print("1")
    pc2m = PointCloudToMeshConverter('/work/mech-ai-scratch/ekimara/DeepSDFCode/Deep-SDF-Autodecorder/data/B73-selected')
    print("2")

    # Get the list of trimesh objects
    meshes = pc2m.get_open3Dmeshes()
    print("3")

    # # Example usage
    converter = MeshToSDFConverter(target_directory="./processed_data/train/")
    print("4")
    converter.process_meshes(meshes)
    print("5")