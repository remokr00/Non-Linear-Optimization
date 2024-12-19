import random
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from seaborn import cm


class Butterfly:

    def __init__(self, bounds, maximization):
        #selecting random position for butterfly
        self.position = np.array([np.random.uniform(low=b[0], high=b[1]) for b in bounds])
        print(f"Initialized position: {self.position}")  # Debug stampa la posizione
        if not maximization:
            self.fitness = np.inf
        else:
            self.fitness = -np.inf
        self.fragrance = 0

def update_fragrance(butterfly, a, c):

   # if butterfly.fitness > 0:
    butterfly.fragrance = c * (butterfly.fitness)**a
    #else:
        #butterfly.fragrance = 0  # Or some other appropriate handling

def move_butterfly(butterfly, best_butterfly, r, butterfly_j, butterfly_k, p):
    move_to_best_butterfly = r < p
    butterfly_pos = np.array(butterfly.position)
    best_butterfly_pos = np.array(best_butterfly.position)
    butterfly_j_pos = np.array(butterfly_j.position)
    butterfly_k_pos = np.array(butterfly_k.position)

    if move_to_best_butterfly:
        return butterfly_pos + ((r**2) * best_butterfly_pos - butterfly_pos) * butterfly.fragrance
    else:
        return butterfly_pos + ((r**2) * butterfly_j_pos - butterfly_k_pos) * butterfly.fragrance


def update_a(iteration, max_iter, initial_a, final_a=0.9):
    # Linear interpolation from initial_a to final_a over max_iter iterations
    return initial_a + ((final_a - initial_a) / max_iter) * iteration

def format_fitness(fitness):

    rounded_fitness = round(fitness, 3)
    if abs(rounded_fitness) < 0.001:
        return 0
    return rounded_fitness

def changing_to_binary(position):
    #i will use the sigmoid function
    new_position = []
    for i in range(len(position)):
        if position[i] < 0.5:
            new_position.append( 0)
        else:
            new_position.append(1)
    return new_position

def teleport(position, mutation_rate):
    #selecting 3 random index
    indexs = np.random.randint(0, len(position), 3)
    for i in indexs:
        if np.random.random() < mutation_rate:
            position[i] = 1 - position[i]
    return position
def butterfly_optimization_algorithm(objective_function, bounds, num_butterflies=10, max_iter=50, c = 0.4, a = 0.0001, p = 0.6, fixed_nodes = [], fixed_values = [], binary = False, maximization = False):

    #initializing the population
    butterflies = [Butterfly(bounds, maximization) for _ in range(num_butterflies)]

    if binary:
        for butterfly in butterflies:
            butterfly.position = changing_to_binary(butterfly.position)

    #i create an array were to store the result of each itereation
    results = []
    evaluations = 0
    best_position = None  # Memorizza la posizione della migliore farfalla
    first_optimal_position = None  # Memorizza la posizione iniziale del primo ottimo trovato
    ris_for_iteration = []
    best_fitness_history = np.inf  # Inizializza con infinito per garantire che il primo confronto funzioni
    unchanged_count = 0  # Contatore per il numero di iterazioni senza miglioramenti

#for every iteration
    for i in range(max_iter):
        print(i)
        for butterfly in butterflies:
            for idx, node_idx in enumerate(fixed_nodes):
                butterfly.position[node_idx*2-2] = fixed_values[2*idx]   # x coordinate
                butterfly.position[node_idx*2-1] = fixed_values[2*idx+1] # y coordinate

        for butterfly in butterflies:

            #we compute the fitness using the objective function
            current_fitness = objective_function(butterfly.position)
            evaluations += 1

            #print(current_fitness)
            if not maximization and current_fitness < butterfly.fitness:
                if butterfly.fitness == np.inf:  # checking if the fitness is infinite
                    butterfly.initial_optimal_position = np.copy(butterfly.position)  # Ssving the intial position
                butterfly.fitness = current_fitness #update the fitness
                update_fragrance(butterfly, a, c)
                evaluations += 1
            elif maximization and current_fitness > butterfly.fitness:
                if butterfly.fitness == -np.inf:
                    butterfly.initial_optimal_position = np.copy(butterfly.position)
                butterfly.fitness = current_fitness
                update_fragrance(butterfly, a, c)
                evaluations += 1



        if not maximization:
            best_butterfly = min(butterflies, key=lambda b: b.fitness)
        else:
            best_butterfly = max(butterflies, key=lambda b: b.fitness)

        results.append(best_butterfly.fitness)

        if best_butterfly.fitness < best_fitness_history:
            best_fitness_history = best_butterfly.fitness
            best_position = best_butterfly.position
            if first_optimal_position is None:
                first_optimal_position = best_butterfly.initial_optimal_position
            unchanged_count = 0
        else:
            unchanged_count += 1

        # Condizione di arresto per mancanza di miglioramenti
        """if unchanged_count >= 50:
           break

            # Condizione di arresto: interrompi se non ci sono cambiamenti per 10 iterazioni
        if unchanged_count >= 50:
            print(f"Stop: Optimal value unchanged for {unchanged_count} iterations.")
            break"""

        for butterfly in butterflies:

            r = np.random.random()
            #selecting the two random batterfluy for the global search
            j = np.random.randint(low=0, high=len(butterflies) - 1)
            k = np.random.randint(low=0, high=len(butterflies) - 1)

            butterfly_j = butterflies[j]
            butterfly_k = butterflies[k]

            butterfly.position = move_butterfly(butterfly, best_butterfly, r, butterfly_j, butterfly_k, p)


            for i in range(len(butterfly.position)):
                if butterfly.position[i] < bounds[i][0]:  # Controlla il limite inferiore
                    butterfly.position[i] = bounds[i][0]
                elif butterfly.position[i] > bounds[i][1]:  # Controlla il limite superiore
                    butterfly.position[i] = bounds[i][1]

            #if len(fixed_nodes) != 0:
            if binary:

                butterfly.position = changing_to_binary(butterfly.position)
                butterfly.position = teleport(butterfly.position, 0.4)



        a = update_a(i, max_iter, a, final_a=0.9)

        ris_for_iteration.append(best_butterfly.fitness)
        print(f"Best fitness for iteration {i}: {format_fitness(best_butterfly.fitness)}")



    return results, evaluations, ris_for_iteration, best_position


#------------------------------------------------------------





def rosenbrock_function(x):
    a = 1
    b = 100
    return  b*(x[1] - x[0]**2)**2 + (a - x[0])**2

#Funziona
def objective_function(x): #funziona
    return sum(x**2)

#Funziona
def sphere_function(x): #funziona
    return sum(xi**2 for xi in x)

def booth_function(X):
    x = X[0]
    y = X[1]
    term1 = (x + 2*y - 7)**2
    term2 = (2*x + y - 5)**2
    return term1 + term2

def beale_function(x):

    a = x[0]
    b = x[1]

    term1 = (1.5 - a + (a * b))**2
    print('term1: ', term1)
    term2 = (2.25 - a + (a * b**2))**2
    print('term2: ', term2)
    term3 = (2.625 - a + (a * b**3))**2
    print('term3: ', term3)
    return term1 + term2 + term3


#Funziona
def rastrigin_function(x, A=10):
    n = len(x)
    return A * n + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)

#si avvicina
def levy_function(X):
    x = X[0]
    y = X[1]
    term1 = np.sin(3 * np.pi * x) ** 2
    term2 = (x - 1) ** 2 * (1 + np.sin(3 * np.pi * y) ** 2)
    term3 = (y - 1) ** 2 * (1 + np.sin(2 * np.pi * y) ** 2)
    return term1 + term2 + term3

#funziona
def ackley_function(x, a=20, b=0.2, c=2*np.pi):
    n = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
    cos_term = -np.exp(np.sum(np.cos(c * x) / n))
    return sum_sq_term + cos_term + a + np.exp(1)

def matyas_function(X):
    x = X[0]
    y = X[1]
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

def goldstein_price_function(X):
    x = X[0]
    y = X[1]
    part1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    part2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return part1 * part2








