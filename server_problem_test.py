# This Python file contains the code for solving the server resource allocation problem.
# First, I define all the variables and fixed values required for the problem.
# Then, I define the constraints and the objective function.
import numpy as np
from Butterfly_OA import butterfly_optimization_algorithm as boa
from pylab import *

# Types of requests that can be made by clients.
# A request is a tuple of 4 values: (request_type, request_duration, request_cpu, request_gains).
# For example, I have defined 3 types of requests as follows:
# Request type A: (A, 2, 3, 10)
# Request type B: (B, 3, 2, 20)
# Request type C: (C, 1, 1, 5)

# Define the request list.
requests = [
    ('A', 2, 3, 10),
    ('B', 3, 12, 20),
    ('C', 1, 17, 5)
]

# Now that the request list is defined, I can define N as the number of requests.
N = len(requests)

# I consider a time interval of 10 units for the problem.
T = 30

# I also define the number of servers available for the problem, considering 4 servers.
S = 5

# Each server has a maximum capacity of 10 units of CPU.
server_capacity = 100

# I define the decision variable for the problem as a 3D matrix where
# x[i][t][j] = 1 if server j is assigned to request i at time t,
# x[i][t][j] = 0 otherwise.
X = [[[0 for j in range(S)] for t in range(T)] for i in range(N)] 

# A queue is needed to store requests that are not currently being processed.
queue = [] 

# It's useful to store the queue from the previous time step.
previous_queue = [0]*N

# The queue has a maximum capacity of 10 requests.
queue_capacity = 30

# I define a penalty for requests that are not currently being processed,
# with different penalties for each request type.
queue_penalty = {
    'A': 0.2,
    'B': 0.3,
    'C': 0.1
}

# Rejected requests are those that cannot be processed now or in the future,
# and they also incur a penalty.
rejected_requests = []

# The penalty for rejected requests, differing by request type.
rejected_penalty = {
    'A': 0.5,
    'B': 0.6,
    'C': 0.4
}

# To evaluate the queue and rejected requests, I create a variable new_requests
# as a 2D matrix with dimensions N x T.
np.random.seed(42)
lam = np.array([5, 3, 1])[:, np.newaxis]  # Lambda values for each request type

# Generate new requests for each type over each time interval

new_requests = np.random.poisson(lam=lam, size=(N, T))


def plot_new_requests(new_requests, request_labels):
    """
    Plots the new requests as a heatmap.

    Parameters:
        new_requests (numpy.ndarray): The matrix of new requests where rows are request types and columns are time intervals.
        request_labels (list): Labels for each request type.
    """
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and an axes.
    cax = ax.matshow(new_requests, cmap='viridis')  # Plot the matrix.

    plt.title('New Requests Over Time')
    plt.xlabel('Time Interval')
    plt.ylabel('Request Type')

    # Add color bar to explain the color encoding
    fig.colorbar(cax)

    # Set tick labels for the y-axis
    ax.set_yticks(np.arange(len(request_labels)))
    ax.set_yticklabels(request_labels)

    # Optionally set x-tick labels if there are specific labels for time intervals
    ax.set_xticks(np.arange(new_requests.shape[1]))
    ax.set_xticklabels(np.arange(1, new_requests.shape[1] + 1))

    plt.show()

plot_new_requests(new_requests, [request[0] for request in requests])
# The objective function aims to maximize the total gains from processing requests,
# considering the penalties for queuing and rejecting requests.
def objective_function(X):
    global requests, queue_penalty, rejected_penalty, new_requests
    #X is a vector of 1s and 0s, representing the assignment of servers to requests.
    #i will convert it into a matrix NxTxS
    X = np.array(X).reshape(N, T, S)
    total_profit = 0
    for i in range(N):
        for t in range(T):
            # Revenue from processing requests
            request_type = ''
            for j in range(S):
                request_type, _, _, profit_per_request = requests[i]
                total_profit += profit_per_request * X[i][t][j]
            # Penalty for queuing requests
            queue_for_request = request_in_queue(X, t, new_requests)
            total_profit -= queue_penalty[request_type] * queue_for_request
            # Penalty for rejecting requests
            rejected_for_request = request_rejected(X, t, new_requests)
            total_profit -= rejected_penalty[request_type] * rejected_for_request
    #adding penalties for costraints violations
    for t in range(T):
        for j in range(S):
            total_profit -= server_capacity_constraint(X, new_requests, server_capacity, t, j) **2
            total_profit -= queue_consistency_constraint(X, t, new_requests, previous_queue, queue_capacity)**2
    for t in range(T):
        total_profit -= queue_capacity_constraint(queue, queue_capacity)**2


    return total_profit

# Function to calculate the number of rejected requests.
def request_in_queue(X, t, new_requests):
    global previous_queue
    queue_count = 0
    for i in range(N):
        requested = new_requests[i][t]
        processed = sum(X[i][t][j] for j in range(S))
        if processed < requested:
            queue_count += (requested - processed)
            previous_queue[i] += (requested - processed)  # Update the previous queue for the next time step
    return queue_count

def request_rejected(X, t, new_requests):
    if t == 0:
        return 0  # non ci sono richieste rifiutate al tempo 0
    rejected_count = 0
    for i in range(N):
        requested = new_requests[i][t]
        processed = sum(X[i][t][j] for j in range(S))
        queued = previous_queue[i] if len(previous_queue) > i else 0
        total_managed = processed + queued
        if total_managed < requested:
            rejected_count += (requested - total_managed)
    return rejected_count


# Constraint: The CPU capacity of each server must not be exceeded.
def server_capacity_constraint(X, requests, server_capacity, t, j):
    sum_cpu = sum(requests[i][2] * X[i][t][j] for i in range(N))
    return np.maximum(0, sum_cpu - server_capacity)

# Constraint: The queue must be within its capacity limits at all times.
def queue_capacity_constraint(queue, queue_capacity):
    return np.maximum(0, len(queue) - queue_capacity)

# Constraint for queue consistency across time steps.
def queue_consistency_constraint(X, t, new_requests, previous_queue, queue_capacity):
    total_queue = 0
    for i in range(N):
        if t == 0:
            total_queue += new_requests[i][t] - sum(X[i][t][j] for j in range(S))
        else:
            total_queue += len(previous_queue) + new_requests[i][t] - sum(X[i][t][j] for j in range(S))
    return np.maximum(0, total_queue - queue_capacity)


# Auxiliary functions for problem analysis.
def check_server_status(X):
    for i in range(N):
        for t in range(T):
            for j in range(S):
                print(f"X[{i}][{t}][{j}] = {X[i][t][j]}")

# Example execution with all servers assigned to every request.
for i in range(N):
    for t in range(T):
        for j in range(S):
            X[i][t][j] = 1

check_server_status(X)

# Example new requests for testing.
new_requests_test = [
    [3, 1, 4],  # Type A requests at time 0, 1, and 2
    [2, 3, 70],  # Type B requests
    [8, 1, 92]   # Type C requests
]

print(queue_consistency_constraint(X, 2, new_requests, previous_queue, queue_capacity))

bounds = [(0, 1)] * (N * T * S)
max_iteration = 1000
a = 0.0001

[ris_1, eval_1, iter_ris_1, optimum_1] = boa(objective_function, bounds, num_butterflies=30,  max_iter=max_iteration, a=a,  binary=True, maximization= True)
[ris_2, eval_2, iter_ris_2, optimum_2] = boa(objective_function, bounds, num_butterflies=50,  max_iter=max_iteration, a=a, binary=True, maximization= True)
[ris_3, eval_3, iter_ris_3, optimum_3] = boa(objective_function, bounds, num_butterflies=100,  max_iter=max_iteration, a=a,  binary=True, maximization= True)

print('Iteration \t Best Fitness \t Num Evaluations \t Num Iterations')
print(f'First Run \t {iter_ris_1[-1]:.3f} \t \t \t{eval_1} \t\t\t {len(iter_ris_1)}')
print(f'Second Run \t {iter_ris_2[-1]:.3f} \t \t \t{eval_2} \t\t\t {len(iter_ris_2)}')
print(f'Third Run \t {iter_ris_3[-1]:.3f} \t \t \t{eval_3} \t\t\t  {len(iter_ris_3)}')
print(optimum_1)
print(optimum_2)
print(optimum_3)

def draw_optimization_eval(evaluations, results, num_runs, colors):
    for i in range(len(evaluations)):
        evaluation = np.arange(0, evaluations[i], 1)  # Ensure evaluation has the right length
        # Check and adjust the length of results to match evaluation
        results[i] = results[i][:evaluations[i]] if len(results[i]) > evaluations[i] else np.pad(results[i], (0, evaluations[i] - len(results[i])), 'edge')
        plt.plot(evaluation, results[i], label=num_runs[i], color=colors[i])
    plt.title('Optimization Performance Comparison')
    plt.xlabel('Evaluations')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def draw_optimization_iterations(iterations, results, num_runs, colors):
    for i in range(len(iterations)):
        # Ensure the x-axis matches the number of iterations correctly
        iterations_array = np.arange(0, len(results[i]))
        plt.plot(iterations_array, results[i], label=num_runs[i], color=colors[i])

    plt.title('Optimization Performance Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_length_for_scheme(results, evaluation):
    if len(results) < evaluation:
        last_value = results[-1] if results else np.inf # i fill the results array to not raise an error due to different length
        results.extend([last_value] * (evaluation - len(results)))

    return results

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