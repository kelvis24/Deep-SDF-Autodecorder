import trimesh
import torch

def get_latent_vector(mesh, decoder, num_points=15000, latent_size=256, init_std=0.01, lr=5e-4, iterations=200):
    # Load the mesh
    # mesh = trimesh.load(mesh_filepath, force='mesh')

    # Sample points on the surface of the mesh
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=num_points)

    # Convert points to torch tensor
    points_tensor = torch.tensor(points).float().cuda()

    # Initialize latent vector
    latent = torch.ones(1, latent_size).normal_(mean=0, std=init_std).cuda()
    latent.requires_grad = True

    # Set optimizer and loss
    optimizer = torch.optim.Adam([latent], lr=lr)
    loss_l1 = torch.nn.L1Loss()
    minT, maxT = -0.1, 0.1  # clamp

    for it in range(iterations):
        decoder.eval()

        # Clamp the ground truth signed distance function (SDF)
        sdf_gt = torch.clamp(torch.tensor(sdf).unsqueeze(1).cuda(), minT, maxT)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_points, -1)
        inputs = torch.cat([latent_inputs, points_tensor], 1)
        pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, minT, maxT)

        # Compute L1 loss
        loss = loss_l1(pred_sdf, sdf_gt)

        # L2 regularization
        loss += 1e-4 * torch.mean(latent.pow(2))

        # Backpropagation
        loss.backward()
        optimizer.step()

        print('[%d/%d] Loss: %.5f' % (it+1, iterations, loss.item()))

    # Return the learned latent vector
    return latent.detach().cpu().numpy()

# Example usage:
latent_vector = get_latent_vector(mesh=open3d_mesh_to_trimesh(meshes[0]), decoder=decoder)
print("Learned latent vector:", latent_vector)
