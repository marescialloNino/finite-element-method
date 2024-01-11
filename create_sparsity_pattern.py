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

def stiff_mat(topol, coord, H):
    # Assigning topology and coordinates to local variables
    T = topol
    C = coord
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

        # Calculating coefficients a, b, c for each node of the element
        a = [x[j] * y[m] - x[m] * y[j], 
             x[m] * y[i] - x[i] * y[m], 
             x[i] * y[j] - x[j] * y[i]]
        b = [y[j] - y[m], y[m] - y[i], y[i] - y[j]]
        c = [x[m] - x[j], x[i] - x[m], x[j] - x[i]]

        # Initializing matrices b_mat and c_mat for basis functions
        b_mat = np.zeros((3, 3))
        c_mat = np.zeros((3, 3))

        # Calculating the entries of the b_mat and c_mat matrices
        for i in range(3):
            for j in range(3):
                b_mat[i, j] = b[i] * b[j]
                c_mat[i, j] = c[i] * c[j]

        # Computing the local stiffness matrix Hloc
        Hloc = (1 / (4 * delta)) * (b_mat + c_mat)

        # Assembling the local stiffness matrix into the global stiffness matrix H
        for i in range(3):
            row = T[k, i] - 1  # Again, adjusting for 0-based indexing
            for j in range(3):
                col = T[k, j] - 1
                H[row, col] += Hloc[i, j]

    # Returning the global stiffness matrix H and the last computed area delta
    return H, delta


H , _ = stiff_mat(topol, coord, H)


plt.spy(H, markersize=1)  # Plot the sparsity pattern of H
plt.title('Sparsity Pattern')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()