import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from fem_solver import solve_steady_state,stiff_and_mass_matrixes, compute_error

ref_grid = pd.read_csv('./meshes/refgrid.csv', sep="\s+", header=None).to_numpy()
delta_t = 0.02
t_max = 10
theta= 0.5

# Initialize lists to store residuals for each mesh and preconditioner
residuals_cho_meshes = []
residuals_jac_meshes = []
error_norms_cho = []
error_norms_jac = []
mesh_size = []
for mesh in range(5):
    print('------RESULTS FOR MESH ', str(mesh),' --------')

    # Load the data from the files and adjust element index for 0-based indexing
    topol = pd.read_csv(f'./meshes/mesh{mesh}/mesh{mesh}.topol', sep="\s+", header=None).to_numpy() 
    topol -= 1
    coord = pd.read_csv(f'./meshes/mesh{mesh}/mesh{mesh}.coord', sep="\s+", header=None).to_numpy()
    bound = pd.read_csv(f'./meshes/mesh{mesh}/mesh{mesh}.bound', sep="\s+", header=None).to_numpy()
    bound[:, 0] -= 1
    trace = pd.read_csv(f'./meshes/mesh{mesh}/mesh{mesh}.trace', sep="\s+", header=None).to_numpy()
    trace[:, 0] -= 1
    track = pd.read_csv(f'./meshes/mesh{mesh}/mesh{mesh}.track', sep="\s+", header=None).to_numpy()
    track[:, 0] -= 1

    H , _ , f = stiff_and_mass_matrixes(topol, coord)

    u0 = np.zeros(H.shape[0])

    u_chol, res_chol,u_jac, res_jac = solve_steady_state(H,bound)

    residuals_cho_meshes.append(res_chol)
    residuals_jac_meshes.append(res_jac)
    
    # Store the residuals for each preconditioner and mesh

    epsilon_cho = compute_error(u_chol,coord,ref_grid,f)
    error_norms_cho.append(epsilon_cho)

    epsilon_jac = compute_error(u_jac,coord,ref_grid,f)
    error_norms_jac.append(epsilon_jac)

    print(f'error norm for cholesky mesh {mesh}: ', epsilon_cho)
    print(f'error norm for jacobi mesh {mesh}: ', epsilon_jac)

    # plot the stationary solutions 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Creating a triangulation object for the plot
    triang = mtri.Triangulation(coord[:, 0], coord[:, 1], topol)

    # Plotting the surface
    surf = ax.plot_trisurf(triang, u_chol, cmap='viridis', edgecolor='none')

    # Add color bar
    fig.colorbar(surf)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('u')

    plt.savefig(f'./plots/stationary_solution_mesh{mesh}.png')
    plt.close() 

    # Assuming topol and coord are numpy arrays
    # topol - n x 3 array, where each row represents a triangle with node indices
    # coord - m x 2 array, where each row represents x and y coordinates of a node

    max_length = 0  # To keep track of the longest side

    for triangle in topol:  # [1,2,3]  -->   [coord[1],coord[2],coord[3]]
        # Get the coordinates of the triangle's vertices
        coords = [coord[node] for node in triangle]

        # Calculate the lengths of each side of the triangle
        side_lengths = [
            np.linalg.norm(coords[0] - coords[1]),
            np.linalg.norm(coords[1] - coords[2]),
            np.linalg.norm(coords[2] - coords[0])
        ]

        # Update max_length if a longer side is found
        max_length = max(max_length, max(side_lengths))
    mesh_size.append(max_length)
    print("Length of the longest side in the mesh:", max_length)

# calculate r_k
r=[0]

for i in range(4):
    r_k = (error_norms_cho[i]/error_norms_cho[i+1])*((mesh_size[i+1]/mesh_size[i])**2)
    r.append(r_k)

print(r)
# Plotting the residuals for each mesh with Cholesky preconditioner
plt.figure(figsize=(10, 6))
for i, residuals in enumerate(residuals_cho_meshes):
    plt.plot(residuals, 'o-' ,mfc='none',label=f'Mesh {i} - Cholesky')
plt.ylabel('Residual norm')
plt.xlabel('Iteration number')
plt.legend()
plt.yscale('log')
plt.title('Convergence Profile with Cholesky Preconditioner')
plt.show()

# Plotting the residuals for each mesh with Jacobi preconditioner
plt.figure(figsize=(10, 6))
for i, residuals in enumerate(residuals_jac_meshes):
    plt.plot(residuals,'o-' ,mfc='none',label=f'Mesh {i} - Jacobi')
plt.ylabel('Residual norm')
plt.xlabel('Iteration number')
plt.legend()
plt.yscale('log')
plt.title('Convergence Profile with Jacobi Preconditioner')
plt.show()