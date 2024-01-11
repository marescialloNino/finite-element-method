import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import ilupp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

# File paths for the uploaded files
topol_file = './meshes/mesh0/mesh0.topol'
coord_file = './meshes/mesh0/mesh0.coord'
bound_file = './meshes/mesh0/mesh0.bound'

# Load the data from the files
topol = pd.read_csv(topol_file, sep="\s+", header=None).to_numpy()
coord = pd.read_csv(coord_file, sep="\s+", header=None).to_numpy()
bound = pd.read_csv(bound_file, sep="\s+", header=None).to_numpy()

n = len(coord)
H = sp.csr_matrix((n,n))
M = sp.csr_matrix((n,n))

def stiff_and_mass_matrixes(topol, coord):

    # Assigning topology and coordinates to local variables
    T = topol
    C = coord
    
   

    n = len(coord)
    H = sp.csr_matrix((n,n))
    M = sp.csr_matrix((n,n))
    f = np.zeros(n)
    
    x = C[:, 0]  # Extracting x-coordinates
    y = C[:, 1]  # Extracting y-coordinates

    # Looping over all elements in the topology matrix
    for k in range(len(T)):
        # Adjusting for 0-based indexing (Python) from 1-based indexing (MATLAB)
        i, j, m = T[k] - 1  

        # Calculating the area of the triangle (element)
        delta = 0.5 * np.linalg.det([[1, x[i], y[i]], 
                                     [1, x[j], y[j]], 
                                     [1, x[m], y[m]]])
        
        # Distribute the area to the nodes, each time the node k appears
        # the value delta/3 is summed to the nodal area of node i
        f[i] += delta / 3
        f[j] += delta / 3
        f[m] += delta / 3

        # Calculating coefficients a, b, c for each node of the element
        a = [x[j] * y[m] - x[m] * y[j], 
             x[m] * y[i] - x[i] * y[m], 
             x[i] * y[j] - x[j] * y[i]]
        b = [y[j] - y[m], y[m] - y[i], y[i] - y[j]]
        c = [x[m] - x[j], x[i] - x[m], x[j] - x[i]]

        
        # Initializing matrices b_mat and c_mat for basis functions
        b_mat = np.zeros((3, 3))
        c_mat = np.zeros((3, 3))
        m_mat = np.ones((3, 3))
        np.fill_diagonal(m_mat, 2)  # Set diagonal elements to 2


        # Calculating the entries of the b_mat and c_mat matrices
        for i in range(3):
            for j in range(3):
                b_mat[i, j] = b[i] * b[j]
                c_mat[i, j] = c[i] * c[j]

        # Computing the local stiffness matrix Hloc
        Hloc = (1 / (4 * delta)) * (b_mat + c_mat)
        # Computing the local mass matrix Mloc
        Mloc = (delta / 12) * m_mat

        # Assembling the local stiffness and mass matrix into the global stiffness
        # matrix H and global mass matrix M
        for i in range(3):
            row = T[k, i] - 1  # Again, adjusting for 0-based indexing
            for j in range(3):
                col = T[k, j] - 1
                H[row, col] += Hloc[i, j]
                M[row, col] += Mloc[i, j]

    # Returning the global stiffness matrix H and the last computed area delta
    return H, M, f


H , M , f = stiff_and_mass_matrixes(topol, coord)

plt.spy(H, markersize=1)  # Plot the sparsity pattern of H
plt.title('Sparsity Pattern')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()



def enforce_dirichlet_conditions(K1,K2, bound, u0):
    nBound = bound.shape[0]
    H = K1.copy()
    rhs = K2 @ u0

    for i in range(nBound):
        # Set all the Dirichlet rows in H to zero
        H[int(bound[i, 0]), :] = 0
        # Update the right-hand side for the Dirichlet boundary
        column_to_subtract = H[:, int(bound[i, 0])].toarray().flatten() * bound[i, 1]
        rhs -= column_to_subtract
        # set Dirichlet columns in H to zero
        H[:, int(bound[i, 0])] = 0
        # Set diagonal term to 1 for Dirichlet nodes in H
        H[int(bound[i, 0]), int(bound[i, 0])] = 1
        # Set rhs to dirichlet condition value in the bound file
        rhs[int(bound[i, 0])] = bound[i, 1]      

    return H, rhs

def solve_system(K1, K2, bound, u0, tmax, deltat):
    t = 0
    u = u0.copy()
    U=[]
    flags=[]
    while t <= tmax  :
        H, rhs = enforce_dirichlet_conditions(K1, K2, bound, u)
        
        M = ilupp.IChol0Preconditioner(H)
        u,flag=sp.linalg.cg(H,rhs,u0,tol=1e-08,maxiter=100,M=M)
        U.append(u)
        flags.append(flag)
        t += deltat
    return U,flags

delta_t = 0.02
theta= 0.5
K1 = (M/delta_t) + (theta * H)
K2 = (M/delta_t) - ((1-theta) * H)
u0 = np.zeros(K1.shape[0])
U,iter = solve_system(K1,K2,bound,u0,10,delta_t)

print(len(U),iter)



# Example Data: Replace these with your actual data
 # Y-coordinates of nodes
triangles = topol - 1 # Connectivity of nodes forming triangles
solution = U[-1]  # Solution at each node
x_coords = coord[:, 0]  # Extracting x-coordinates
y_coords = coord[:, 1]  # Extracting y-coordinates
# Creating the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Creating a triangulation object for the plot
triang = mtri.Triangulation(x_coords, y_coords, triangles)

# Plotting the surface
surf = ax.plot_trisurf(triang, solution, cmap='viridis', edgecolor='none')

# Add color bar
fig.colorbar(surf)

# Setting labels (optional)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Solution Value')

plt.show()