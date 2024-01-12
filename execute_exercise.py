import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import ilupp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from create_matrixes import stiff_and_mass_matrixes, solve_system, compute_error
import time

# global variables
ref_grid = pd.read_csv('./meshes/refgrid.csv', sep="\s+", header=None).to_numpy()
delta_t = 0.02
t_max = 10
theta= 0.5

for mesh in range(3):
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

    H , M , f = stiff_and_mass_matrixes(topol, coord)

    plt.spy(H, markersize=1)  # Plot the sparsity pattern of H
    plt.title('Sparsity Pattern')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    # Save the plot to a file instead of showing it
    plt.savefig(f'./plots/sparsity_pattern_mesh{mesh}.png')
    plt.close()  # Close the plot to free up memory

    
    K1 = (M/delta_t) + (theta * H)
    K2 = (M/delta_t) - ((1-theta) * H)
    u0 = np.zeros(K1.shape[0])
    start = time.time()
    U,flags = solve_system(K1,K2,bound,u0,t_max,delta_t)
    print(f'solve system for mesh {mesh} cpu time:',time.time()-start)

    epsilon = compute_error(U[-1],coord,ref_grid,f)

    print(f'error norm for mesh {mesh}: ', epsilon)

    # temperature on the boundary gammaN at different times
    node_index_trace = trace[:,0].astype(int)
    arc_length = trace[:, 1]
    times_trace = [125,250,375,500]
    trace_solutions=[]

    for i in times_trace:
        sol = U[i]
        trace_u=[]
        for index in node_index_trace:
            trace_u.append(sol[index])
        trace_solutions.append(trace_u)

    plt.plot(arc_length, trace_solutions[0])
    plt.plot(arc_length, trace_solutions[1])
    plt.plot(arc_length, trace_solutions[2])
    plt.plot(arc_length, trace_solutions[3])

    plt.title('Trace profile')
    plt.xlabel('Arc length')
    plt.ylabel('u')
    plt.legend(['t=2.5s','t=5.0s','t=7.5s','t=10s'])
    plt.savefig(f'./plots/trace_profile_mesh{mesh}.png')
    plt.close() 

    # solutions at track points p1, p2, p3 over the time range
    times = np.arange(0, 10.02, 0.02)
    track_solutions = []
    for i in track:
        
        node_index = i 
        sol = []
        for solution in U:
            sol.append(solution[node_index])
            
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
    plt.savefig(f'./plots/track_nodes_mesh{mesh}.png')
    plt.close() 

    # Values of tracking points at final time
    print('solution for P1 at tmax: ',track_solutions[0][-1])
    print('solution for P2 at tmax: ',track_solutions[1][-1])
    print('solution for P3 at tmax: ',track_solutions[2][-1])

    # plot the solutions at final time
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Creating a triangulation object for the plot
    triang = mtri.Triangulation(coord[:, 0], coord[:, 1], topol)

    # Plotting the surface
    surf = ax.plot_trisurf(triang, U[-1], cmap='viridis', edgecolor='none')

    # Add color bar
    fig.colorbar(surf)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('u')

    plt.savefig(f'./plots/final_solution_mesh{mesh}.png')
    plt.close() 
