import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import ilupp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from create_matrixes import stiff_and_mass_matrixes, solve_system, compute_error
import time

ref_grid_file = './meshes/refgrid.csv'


# File paths for the uploaded files
topol_file = './meshes/mesh1/mesh1.topol'
coord_file = './meshes/mesh1/mesh1.coord'
bound_file = './meshes/mesh1/mesh1.bound'
trace_file = './meshes/mesh1/mesh1.trace'
track_file = './meshes/mesh1/mesh1.track'

# Load the data from the files and adjust element index for 0-based indexing
topol = pd.read_csv(topol_file, sep="\s+", header=None).to_numpy() 
topol -= 1
coord = pd.read_csv(coord_file, sep="\s+", header=None).to_numpy()
bound = pd.read_csv(bound_file, sep="\s+", header=None).to_numpy()
bound[:, 0] -= 1
trace = pd.read_csv(trace_file, sep="\s+", header=None).to_numpy()
trace[:, 0] -= 1
track = pd.read_csv(track_file, sep="\s+", header=None).to_numpy()
track[:, 0] -= 1
ref_grid = pd.read_csv(ref_grid_file, delim_whitespace=True, header=None).to_numpy()



n = len(coord)
H = sp.csr_matrix((n,n))
M = sp.csr_matrix((n,n))

H , M , f = stiff_and_mass_matrixes(topol, coord)

plt.spy(H, markersize=1)  # Plot the sparsity pattern of H
plt.title('Sparsity Pattern')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()


delta_t = 0.02
theta= 0.5
K1 = (M/delta_t) + (theta * H)
K2 = (M/delta_t) - ((1-theta) * H)
u0 = np.zeros(K1.shape[0])
start = time.time()
U,iter = solve_system(K1,K2,bound,u0,10,delta_t)
print('solve sistem cpu time:',time.time()-start)
epsilon = compute_error(U[-1],coord,ref_grid,f)

print(epsilon)

# temperature on the boundary gammaN at different times

node_index_trace = trace[:,0]
node_index_trace = node_index_trace.astype(int)
arc_length = trace[:, 1]
times_trace = [124,249,374,499]
trace_solutions=[]

for time in times_trace:
    sol = U[time]
    trace_u=[]
    for index in node_index_trace:
        trace_u.append(sol[index])
    trace_solutions.append(trace_u)

plt.plot(arc_length, trace_solutions[0])
plt.plot(arc_length, trace_solutions[1])
plt.plot(arc_length, trace_solutions[2])
plt.plot(arc_length, trace_solutions[3])

# Add titles and labels
plt.title('Trace profile')
plt.xlabel('Arc length')
plt.ylabel('u')
plt.legend(['t=2.5s','t=5.0s','t=7.5s','t=10s'])
plt.show()

times = np.arange(0, 10.02, 0.02)
track_solutions = []
for i in track:
    
    node_index = i 
    sol = []
    for solution in U:
        sol.append(solution[node_index])
        # Create the plot
    track_solutions.append(sol)    

plt.plot(times, track_solutions[0])
plt.plot(times, track_solutions[1])
plt.plot(times, track_solutions[2])
# Add titles and labels
plt.title('Node Values Over Time')
plt.xlabel('Time')
plt.ylabel('u')
plt.legend(['P1','P2','P3'])
plt.grid(True)

# Show the plot
plt.show()

# Values of tracking points at final time
print('solution for P1 at tmax: ',track_solutions[0][-1])
print('solution for P2 at tmax: ',track_solutions[1][-1])
print('solution for P3 at tmax: ',track_solutions[2][-1])

""" error_last_time_step = compute_error(U[-1], x_coords, y_coords, xRef, yRef, uRef)
print("Error at the last time step:", error_last_time_step) """

triangles = topol  # Connectivity of nodes forming triangles
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