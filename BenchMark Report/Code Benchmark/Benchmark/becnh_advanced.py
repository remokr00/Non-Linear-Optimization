
from Butterfly_OA import butterfly_optimization_algorithm as boa
from pylab import *

# Initialization of variables for a truss structure composed of 10 bars
n_bars = 36  # Number of bars
n_nodes = 15  # Number of nodes
L = 0.915  # Characteristic length (m)
force_magnitude = 1.0e5
E = 68.9e9  # Elasticity of the material in Pascals
rho = 2700  # Density of the material in kg/m^3
sigma_max = 2.76e8  # Maximum allowable stress in Pascals
displacements = np.full(2*n_nodes, 1e-5)
fixed_nodes = [0,1,6,7]
# Area for shape optimization (constant)
area_constant = 1e-3  # in square meters
area = np.full(n_bars, 1e-3)  # Initial cross-sectional area of each bar in square meters

external_forces= np.zeros(2 * n_nodes)  # External forces assumed to be zero initially
external_forces[18] = force_magnitude  # Forza applicata in x per il nodo 10
external_forces[28] = force_magnitude  # Forza applicata in x per il nodo 15
external_forces[29] = -force_magnitude  # Forza applicata in y per il nodo 15


#optional truss problem
nodes_initial = np.array([
    [0, 2*L],
    [L, 2*L],
    [2*L, 2*L],
    [3*L, 2*L],
    [4*L, 2*L],


    [0, L],
    [L, L],
    [2*L, L],
    [3*L, L],
    [4*L, L],

    [0, 0],
    [L, 0],
    [2*L, 0],
    [3*L, 0],
    [4*L, 0],

])

bars = [
    #horizontal
    (0, 1), (1, 2),(2, 3), (3, 4),
    (5, 6), (6, 7),(7, 8), (8, 9),
    (10, 11), (11, 12), (12, 13), (13, 14),

    #diagonal
    (0, 6), (1, 5), (1, 7), (2, 6), (2, 8), (3, 7), (3, 9), (4, 8),
    (5, 11), (6, 10), (6, 12), (7, 11), (7, 13), (8, 12), (8, 14), (9, 13),

    #vertical
    (1, 6), (6, 11),
    (2, 7), (7, 12),
    (3, 8), (8, 13),
    (4, 9), (9, 14),

]

def compute_lengths(nodes, bars):
    return np.array([np.linalg.norm(nodes[j] - nodes[i]) for i, j in bars])






# Compute the lengths of each bar


# Compute the lengths of each bar
lengths = compute_lengths(nodes_initial, bars)

# Objective function to minimize the total mass of the truss structure
def truss_objective_function_area(areas):
    global rho, L, nodes_initial, bars, E, external_forces, sigma_max, fixed_nodes

    K_global = assemble_global_stiffness(nodes_initial, bars, E, areas, fixed_nodes)
    try:
        displacements = equilibrium_constraint(K_global, external_forces)
        max_displacement = np.max(np.abs(displacements))  # Calculate maximum displacement for normalization
        displacements_norm = displacements / max_displacement  # Normalize displacements
        displacement_penalty = displacements_norm**2 # Penalizing large displacements
    except ValueError as e:
        displacement_penalty = np.inf

    forces = calculate_axial_forces(displacements, bars, nodes_initial, areas, E)
    lengths_of_bars = [L]*n_bars  # array of lengths

    stress_penalties = stress_constraints(forces, areas, sigma_max)
    stress_norm = (stress_penalties / sigma_max) ** 2 # Normalize stress penalties
    area_penalties = area_constraints(areas, 1.9635e-5)
    area_norm = (area_penalties / 1.9635e-5) ** 2  # Normalize area penalties

    mass_total = rho * np.sum(areas * lengths_of_bars)
    penalty_total_1 = (np.sum(area_norm))
    penalty_total_2 = (np.sum(displacement_penalty))
    penalty_total_3 = 1e-2 * (np.sum(stress_norm))
    penalty_total = penalty_total_1 + penalty_total_2 + penalty_total_3
    total_cost = mass_total + penalty_total

    return total_cost

def truss_objective_function_coord(coordinates):
    global rho, L, area_constant, bars, E, external_forces, sigma_max, displacements

    areas = np.full(n_bars, area_constant)
    # Adapt coordinates format
    coord_for_k = []

    # i want to have a matrix with the coordinates of the nodes
    for i in range(0, len(coordinates), 2):
        coord_for_k.append([coordinates[i], coordinates[i+1]])

    K_global = assemble_global_stiffness(coord_for_k, bars, E, areas, fixed_nodes)
    try:
        displacements = equilibrium_constraint(K_global, external_forces)
        max_displacement = np.max(np.abs(displacements))  # Calculate maximum displacement for normalization
        displacements_norm = displacements / max_displacement  # Normalize displacements
        displacement_penalty = displacements_norm**2 # Penalizing large displacements
    except ValueError as e:
        displacement_penalty = np.inf


    coord_for_k = np.reshape(coordinates, (-1, 2))
    forces = calculate_axial_forces(displacements, bars, coord_for_k, areas, E)
    lengths_of_bars = [L] * n_bars

    stress_penalties = stress_constraints(forces, areas, sigma_max)
    stress_norm = (stress_penalties / sigma_max) ** 2
    length_constraint = compute_lengths(coord_for_k, bars)
    non_negative_length_constraints_norm =  (non_negative_length_constraints(length_constraint) / np.max(length_constraint)) ** 2
    coord_constraints = non_negative_node_coordinates_constraints(coord_for_k)
    mass_total = rho * area_constant * np.sum(lengths_of_bars)
    penalty_total = np.sum(non_negative_length_constraints_norm) + np.sum(displacement_penalty) + (1e-2 * np.sum(stress_norm)) + np.sum(coord_constraints)

    #if two or more nodes have at leat one same coordinates or are too close to each other, add a large penalty
    #i use a for
    same_coord_penalty = 0
    for i in range(0, len(coord_for_k)):
        for j in range(i+1, len(coord_for_k)):
            if coord_for_k[i][0] == coord_for_k[j][0] and coord_for_k[i][1] == coord_for_k[j][1]:
                same_coord_penalty = 1e10
            elif np.linalg.norm(coord_for_k[i] - coord_for_k[j]) < 0.1:
                same_coord_penalty = 1e10

    over_coord_penalty = 0
    #also if the nodes are over or under the height of the fixed points of the truss, add a large penalty
    for i in range(0, len(coord_for_k)):
        if coord_for_k[i][1] < 0 or coord_for_k[i][1] > 2*L :
            over_coord_penalty = 1e10

    total_cost = mass_total + penalty_total + same_coord_penalty + over_coord_penalty

    return total_cost



#Computes the local stiffness matrix for a bar element in the truss structure.
def local_stiffness_matrix(E, area, L, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    k = (E * area / L) * np.array([
        [c**2, c*s, -c**2, -c*s],
        [c*s, s**2, -c*s, -s**2],
        [-c**2, -c*s, c**2, c*s],
        [-c*s, -s**2, c*s, s**2]
    ])
    return k

#Assembles the global stiffness matrix for the entire truss structure.
def assemble_global_stiffness(nodes, bars, E, areas, fixed_indices):

    global nodes_initial
    n = len(nodes_initial)
    K_global = np.zeros((2 * n, 2 * n))
    for idx, (i, j) in enumerate(bars):
        xi, yi = nodes[i]
        xj, yj = nodes[j]

        L = np.linalg.norm([xj - xi, yj - yi])
        theta = np.arctan2(yj - yi, xj - xi)
        k_local = local_stiffness_matrix(E, areas[idx], L, theta)
        indices = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        for ii in range(4):
            for jj in range(4):
                K_global[indices[ii], indices[jj]] += k_local[ii, jj]

    # Apply boundary conditions for fixed nodes (node 1 and node 4 as per problem statement)
    for fi in fixed_indices:
        K_global[fi, :] = K_global[:, fi] = 0
        K_global[fi, fi] = 1e10

    return K_global

def calculate_axial_forces(displacements, bars, nodes, areas, E):
    forces = []
    for (i, j), area in zip(bars, areas):
        L = np.linalg.norm(nodes[j] - nodes[i])
        theta = np.arctan2(nodes[j][1] - nodes[i][1], nodes[j][0] - nodes[i][0])
        c = np.cos(theta)
        s = np.sin(theta)
        # Element stiffness matrix in global coordinates
        k_local = (E * area / L) * np.array([
            [c**2, c*s, -c**2, -c*s],
            [c*s, s**2, -c*s, -s**2],
            [-c**2, -c*s, c**2, c*s],
            [-c*s, -s**2, c*s, s**2]
        ])
        # Indices in the global displacement vector
        index_i = [2*i, 2*i+1]
        index_j = [2*j, 2*j+1]
        disp_vector = np.concatenate((displacements[index_i], displacements[index_j]))
        # Calculate axial force
        force = np.dot(k_local, disp_vector)
        forces.append(force[0])
    return np.array(forces)

def stress_constraints(forces, areas, sigma_max):
    stresses = forces / areas
    # Aumentiamo la penalità se lo stress supera sigma_max
    stress_penalty = np.maximum(0, np.abs(stresses) - sigma_max)
    return stress_penalty


# Constraint ensuring that the area of each bar is not below a minimum value
def area_constraints(area, area_min):
    # Penalità maggiore per area sotto il minimo
    area_penalty = np.maximum(0, area_min - area)
    return area_penalty

# Constraint that ensures the structure is in static equilibrium
def equilibrium_constraint(K_global, external_forces):
    if np.linalg.det(K_global) == 0:
        raise ValueError("Singular matrix, it's impossible to invert")
    displacements = np.linalg.solve(K_global, external_forces)

    return displacements

# Constraint that ensures the length of each bar is non-negative
def non_negative_length_constraints(lengths):
    penalty = np.sum(np.maximum(-lengths, 0))
    return penalty

# Constraint that ensures the coordinates of each node are non-negative
def non_negative_node_coordinates_constraints(nodes):
    penalty = np.sum(np.maximum(-nodes, 0))
    return penalty


# Auxiliar function for drawing ------------------------
def check_length_for_scheme(results, evaluation):
    if len(results) < evaluation:
        last_value = results[-1] if results else np.inf # i fill the results array to not raise an error due to different length
        results.extend([last_value] * (evaluation - len(results)))

    return results

def draw_optimization_eval(evaluations, results, num_runs, colors):
    for i in range (0, len(evaluations)):
        evaluation = np.arange(0, evaluations[i], 1)
        plt.plot(evaluation, results[i], label=num_runs[i], color=colors[i])
    plt.title('Optimization Performance Comparison Optimization')
    plt.xlabel('Evaluations')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def draw_optimization_iterations(iterations, results, num_runs, colors):
    for i in range (0, len(iterations)):
        print(iterations)
        iterations_array = np.arange(0, iterations[i], 1)
        plt.plot(iterations_array, results[i], label=num_runs[i], color=colors[i])
    plt.title('Optimization Performance Comparison Optimization')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_truss_structure(nodes, figure_number):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (i, j) in bars:
        start_node = nodes[i]
        end_node = nodes[j]
        line = plt.Line2D((start_node[0], end_node[0]), (start_node[1], end_node[1]), lw=0.1 * 10, color='grey')
        ax.add_line(line)
    for index, (x, y) in enumerate(nodes):
        if index in [0, 5, 10]:  # Nodi fissi
            ax.plot(x, y, 'ro', markersize=10, marker='>')  # Nodo fisso
        else:
            ax.plot(x, y, 'go', markersize=5)  # Nodo libero
    ax.set_xlim(-0.2, 2 * max(nodes[:, 0]))
    ax.set_ylim(-0.2, 1.1 * max(nodes[:, 1]))
    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_title(f'Optimized Truss Structure {figure_number}')
    plt.show()

# Solving the problems ----------------------------------------
min_area = 1.9635e-5
max_area = 1
bounds = [(min_area, max_area)] * n_bars
max_iteration = 5000
a = 0.0001

[ris_1, eval_1, iter_ris_1, optimum_1] = boa(truss_objective_function_area, bounds, num_butterflies=50,  max_iter=max_iteration, a=a, fixed_nodes=[], fixed_values=[])
[ris_2, eval_2, iter_ris_2, optimum_2]= boa(truss_objective_function_area, bounds, num_butterflies= 100,  max_iter=max_iteration, a=a, fixed_nodes=[], fixed_values=[])
[ris_3, eval_3, iter_ris_3, optimum_3] = boa(truss_objective_function_area, bounds, num_butterflies= 150,  max_iter=max_iteration, a=a, fixed_nodes=[], fixed_values=[])

ris_eval_1 = check_length_for_scheme(ris_1, eval_1)
ris_eval_2 = check_length_for_scheme(ris_2, eval_2)
ris_eval_3 = check_length_for_scheme(ris_3, eval_3)


evaluations = [eval_1, eval_2, eval_3]
results_eval = [ris_eval_1, ris_eval_2, ris_eval_3]
runs = ['First Run', 'Second Run', 'Third Run']
colors = ['red', 'green', 'blue']

draw_optimization_eval(evaluations, results_eval, runs, colors)

iterations = [len(iter_ris_1), len(iter_ris_2), len(iter_ris_3)]
result_iter = [iter_ris_1, iter_ris_2, iter_ris_3]

draw_optimization_iterations(iterations, result_iter, runs, colors)

print('Iteration \t Best Fitness \t Num Evaluations \t Num Iterations')
print(f'First Run \t {iter_ris_1[-1]:.3f} \t \t \t{eval_1} \t\t\t {len(iter_ris_1)}')
print(f'Second Run \t {iter_ris_2[-1]:.3f} \t \t \t{eval_2} \t\t\t {len(iter_ris_2)}')
print(f'Third Run \t {iter_ris_3[-1]:.3f} \t \t \t{eval_3} \t\t\t  {len(iter_ris_3)}')
print(optimum_1)
print(optimum_2)
print(optimum_3)




fig, ax = plt.subplots()
for (i, j), area in zip(bars, optimum_1):
    start_node = nodes_initial[i]
    end_node = nodes_initial[j]
    line = plt.Line2D((start_node[0], end_node[0]), (start_node[1], end_node[1]), lw=area*10000, color='grey', solid_capstyle='round')
    ax.add_line(line)

# Add nodes with different colors for fixed and free nodes
for index, (x, y) in enumerate(nodes_initial):
    if index in [0, 5, 10]:  # Assuming nodes 0 and 3 are fixed
        ax.plot(x, y, 'ro', markersize=10, marker = '>')  # Red color for fixed nodes
    else:
        ax.plot(x, y, 'go', markersize=5)  # Green color for other nodes

# Set limits, labels, and grid
ax.set_xlim(-0.2, 4*L +0.2)
ax.set_ylim(-0.2, 2*L + 0.2)
ax.set_xlabel('X Coordinate (meters)')
ax.set_ylabel('Y Coordinate (meters)')
ax.set_title('Optimized Truss Structure')


# Show the plot
plt.show()

def truss_draw():
    fig, ax = plt.subplots()
    for (i, j), area in zip(bars, optimum_2):
        start_node = nodes_initial[i]
        end_node = nodes_initial[j]
        line = plt.Line2D((start_node[0], end_node[0]), (start_node[1], end_node[1]), lw=area*10000, color='grey', solid_capstyle='round')
        ax.add_line(line)

    # Add nodes with different colors for fixed and free nodes
    for index, (x, y) in enumerate(nodes_initial):
        if index in [0, 5, 10]:  # Assuming nodes 0 and 3 are fixed
            ax.plot(x, y, 'ro', markersize=10, marker = '>')  # Red color for fixed nodes
        else:
            ax.plot(x, y, 'go', markersize=5)  # Green color for other nodes

    # Set limits, labels, and grid
    ax.set_xlim(-0.2, 4*L +0.2)
    ax.set_ylim(-0.2, 2*L + 0.2)
    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_title('Optimized Truss Structure')


    # Show the plot
    plt.show()

fig, ax = plt.subplots()
for (i, j), area in zip(bars, optimum_3):
    start_node = nodes_initial[i]
    end_node = nodes_initial[j]
    line = plt.Line2D((start_node[0], end_node[0]), (start_node[1], end_node[1]), lw=area*10000, color='grey', solid_capstyle='round')
    ax.add_line(line)

# Add nodes with different colors for fixed and free nodes
for index, (x, y) in enumerate(nodes_initial):
    if index in [0, 5, 10]:  # Assuming nodes 0 and 3 are fixed
        ax.plot(x, y, 'ro', markersize=10,  marker = '>')  # Red color for fixed nodes
    else:
        ax.plot(x, y, 'go', markersize=5)  # Green color for other nodes

# Set limits, labels, and grid
ax.set_xlim(-0.2, 4*L +0.2)
ax.set_ylim(-0.2, 2*L + 0.2)
ax.set_xlabel('X Coordinate (meters)')
ax.set_ylabel('Y Coordinate (meters)')
ax.set_title('Optimized Truss Structure')


# Show the plot
plt.show()


fixed_nodes = [1, 6, 11, 15]
fixed_values = [0, 2*L, 0, L, 0, 0, 4*L, 0]
bounds = [(0, 2*L)] * 30
[ris_1, eval_1, iter_ris_1, optimum_1] = boa(truss_objective_function_coord, bounds, num_butterflies=50,  max_iter=max_iteration, a=a, fixed_nodes = fixed_nodes, fixed_values = fixed_values)
[ris_2, eval_2, iter_ris_2, optimum_2]= boa(truss_objective_function_coord, bounds, num_butterflies= 100,  max_iter=max_iteration, a=a, fixed_nodes = fixed_nodes, fixed_values = fixed_values)
[ris_3, eval_3, iter_ris_3, optimum_3] = boa(truss_objective_function_coord, bounds, num_butterflies= 150,  max_iter=max_iteration, a=a,fixed_nodes = fixed_nodes, fixed_values = fixed_values)

print('Iteration \t Best Fitness \t Num Evaluations \t Num Iterations')
print(f'First Run \t {iter_ris_1[-1]:.3f} \t \t \t{eval_1} \t\t\t {len(iter_ris_1)}')
print(f'Second Run \t {iter_ris_2[-1]:.3f} \t \t \t{eval_2} \t\t\t {len(iter_ris_2)}')
print(f'Third Run \t {iter_ris_3[-1]:.3f} \t \t \t{eval_3} \t\t\t  {len(iter_ris_3)}')
print(optimum_1)
print(optimum_2)
print(optimum_3)

ris_eval_1 = check_length_for_scheme(ris_1, eval_1)
ris_eval_2 = check_length_for_scheme(ris_2, eval_2)
ris_eval_3 = check_length_for_scheme(ris_3, eval_3)

evaluations = [eval_1, eval_2, eval_3]
results_eval = [ris_eval_1, ris_eval_2, ris_eval_3]
runs = ['First Run', 'Second Run', 'Third Run']
colors = ['red', 'green', 'blue']

draw_optimization_eval(evaluations, results_eval, runs, colors)

iterations = [len(iter_ris_1), len(iter_ris_2), len(iter_ris_3)]
result_iter = [iter_ris_1, iter_ris_2, iter_ris_3]

draw_optimization_iterations(iterations, result_iter, runs, colors)

nodes_optimized_1 = np.array(optimum_1).reshape(-1, 2)
nodes_optimized_2 = np.array(optimum_2).reshape(-1, 2)
nodes_optimized_3 = np.array(optimum_3).reshape(-1, 2)

# Creazione dei plot per ciascun set di coordinate ottimizzate
plot_truss_structure(nodes_optimized_1, 1)
plot_truss_structure(nodes_optimized_2, 2)
plot_truss_structure(nodes_optimized_3, 3)


