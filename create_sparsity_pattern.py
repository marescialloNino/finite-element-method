import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

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

def stiff_mat(topol, coord):

    # Assigning topology and coordinates to local variables
    T = topol
    C = coord
    
    f = np.zeros(n)

    n = len(coord)
    H = sp.csr_matrix((n,n))
    M = sp.csr_matrix((n,n))
    
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


H , M , _ = stiff_mat(topol, coord)

def enforce_dirichlet_conditions(H, bound):
    nBound = bound.shape[0]

    n = H.shape[0]
    # Initialize the RHS as a zero vector
    rhs = np.zeros(n)

    # Step 1: Set all the Dirichlet rows in H to zero
    for i in range(nBound):
        H[bound[i, 0], :] = 0

    # Step 2: Adjust the RHS for the Dirichlet conditions
    for i in range(nBound):
        # Update the right-hand side for the Dirichlet boundary
        rhs -= H[:, bound[i, 0]] * bound[i, 1]
        H[:, bound[i, 0]] = 0
        rhs[bound[i, 0]] = bound[i, 1]

    # Step 3: Set diagonal term to 1 for Dirichlet nodes in H
    for i in range(nBound):
        H[bound[i, 0], bound[i, 0]] = 1

    return H, rhs
H1 , _ = enforce_dirichlet_conditions(H,bound)

plt.figure(figsize=(10, 10))  # Create a new figure with a specified size
plt.spy(H, markersize=1)  # Plot the sparsity pattern of H
plt.title('Sparsity Pattern')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()

plt.figure(figsize=(10, 10))  # Create a new figure with a specified size
plt.spy(H1, markersize=1)  # Plot the sparsity pattern of H
plt.title('Sparsity Pattern')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()