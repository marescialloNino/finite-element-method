import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import ilupp

def stiff_and_mass_matrixes(topol, coord):

    # Assigning topology and coordinates to local variables
    T = topol
    C = coord
    
    n = coord.shape[0]
    H = sp.csr_matrix((n,n))
    M = sp.csr_matrix((n,n))
    f = np.zeros(n)
    
    x = C[:, 0]  # Extracting x-coordinates
    y = C[:, 1]  # Extracting y-coordinates

    # Looping over all elements in the topology matrix
    for k in range(len(T)):
        # Adjusting for 0-based indexing (Python) from 1-based indexing (MATLAB)
        i, j, m = T[k] - 1  

        # Calculating the area of the triangular element
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
