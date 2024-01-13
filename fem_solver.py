import numpy as np
import scipy.sparse as sp
import ilupp
from scipy.interpolate import LinearNDInterpolator

def stiff_and_mass_matrixes(topol, coord):

    # Assigning topology and coordinates to local variables
    T = topol
    C = coord
    
    n = coord.shape[0]
    H = sp.csr_matrix((n,n)).tolil()
    M = sp.csr_matrix((n,n)).tolil()
    f = np.zeros(n)
    
    x = C[:, 0]  # Extracting x-coordinates
    y = C[:, 1]  # Extracting y-coordinates

    # Looping over all elements in the topology matrix
    for k in range(len(T)):
        
        i, j, m = T[k]  

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
            row = T[k, i]
            for j in range(3):
                col = T[k, j] 
                H[row, col] += Hloc[i, j]
                M[row, col] += Mloc[i, j]

    # Returning the global stiffness matrix H and the last computed area delta
    return H.tocsr(), M.tocsr(), f



def enforce_dirichlet_conditions(K1,K2, bound, u0, current_time):
    nBound = bound.shape[0]
    H = K1.copy().tolil() # Convert to lil_matrix for efficient modifications
    rhs = K2 @ u0
    tmax=10
    for i in range(nBound):
        node_index = int(bound[i, 0])
        boundary_type = bound[i, 1]

        # Determine the boundary value based on the type and current time
        if boundary_type == 0: 
            boundary_value = 0  # Constant zero
        elif boundary_type == 1:
            if current_time <= tmax / 2:
                boundary_value = 2 * current_time / tmax  # Linearly increasing
            else:
                boundary_value = 1  # Constant at 1

        # Set all the Dirichlet rows in H to zero
        H[node_index, :] = 0
        # Update the right-hand side for the Dirichlet boundary
        column_to_subtract = H[:, node_index].toarray().flatten() * boundary_value
        rhs -= column_to_subtract
        # set Dirichlet columns in H to zero
        H[:, node_index] = 0
        # Set diagonal term to 1 for Dirichlet nodes in H
        H[node_index, node_index] = 1
        # Set rhs to dirichlet condition value in the bound file
        rhs[node_index] = boundary_value      
    # return updated matrix K2 and rhs
    return H.tocsr(), rhs


def solve_system(K1, K2, bound, u0, tmax, deltat):
    # takes K1,K2 systemmatrixes and u0 the initial solution
    t = 0
    u = u0.copy()
    U=[]
    flags=[]
    # solve linear system for each timestamp, enforsing boundary condition at every step
    # and ricalculating the preconditioner for the updated K1 with each solution u_k found
    # in the previous step
    
    while t <= tmax :
        H, rhs = enforce_dirichlet_conditions(K1, K2, bound, u, t)      
        M = ilupp.IChol0Preconditioner(H)
        u,flag=sp.linalg.cg(H,rhs,u0,tol=1e-05,maxiter=100,M=M)
        U.append(u)
        flags.append(flag)
        t += deltat
    return U,flags

def solve_steady_state(H, bound):
    A = H.copy().tolil()    
    b = np.zeros(H.shape[0])
    # Impose Dirichlet boundary conditions on A
    for i in range(bound.shape[0]):

        A[int(bound[i, 0]), :] = 0
        b -= (A[:, int(bound[i, 0])].toarray().flatten()) * bound[i, 1]
        A[:, int(bound[i, 0])] = 0
        A[int(bound[i, 0]), int(bound[i, 0])] = 1
        b[int(bound[i, 0])] = bound[i, 1]
    
    A = A.tocsr()
    Mchol = ilupp.IChol0Preconditioner(A) 
    Mjac = sp.diags(1/A.diagonal())

    def store_residuals(A,b,residuals):
            def callback(xk):
                residuals.append(np.linalg.norm(b - (A @ xk)))
            return callback
    
    resvec_cho = []
    callback_cho = store_residuals(A,b,resvec_cho)
    # Solve linear system using Conjugate Gradient method with preconditioner
    solution_cho, _ = sp.linalg.cg(A, b, tol=1e-08, maxiter=1000, M=Mchol, callback = callback_cho)

    resvec_jac = []
    callback_jac = store_residuals(A,b,resvec_jac)
    solution_jac, _ = sp.linalg.cg(A, b, tol=1e-08, maxiter=1000, M=Mjac,callback = callback_jac)

    return solution_cho, resvec_cho, solution_jac, resvec_jac

def compute_error(u, coord, ref_grid, f):

    x_coords = coord[:, 0]
    y_coords = coord[:, 1]
    # Interpolate the reference solution onto the mesh
    interp = LinearNDInterpolator((ref_grid[:,0], ref_grid[:,1]), ref_grid[:,2])

    ref_solution_interpolated = interp(x_coords, y_coords)

    eps_squared = 0
    for i in range(len(u)):
        # sum contribute for each node, using the nodal areas calculated 
        # while creating the stifness matrix
        eps_squared += ((u[i] - ref_solution_interpolated[i])**2)*f[i]
    # Compute the error
    eps = np.sqrt(eps_squared)
    
    return eps
