import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Butterfly_OA import butterfly_optimization_algorithm as BOA

# Parametri fissi
V0 = 299792.458  # km/h
R = 6371  # km
dmin = 2000  # km
dmax = 36000  # km
coord_limit_sat = (-(R + dmax), R + dmax)
coord_limit_orbit = (- (R+dmin), R+dmin)

# Pesi
w1 = 0.5
w2 = 0.5

# Numero di satelliti e stazioni di terra
N = M = 5

# Limiti per le posizioni dei satelliti
bound = [coord_limit_sat] * 2*N



# Genera le coordinate dei satelliti in 2D
#Cs = np.random.uniform(coord_limit_sat[0], coord_limit_sat[1], (N, 2))

# Funzione per generare le coordinate delle stazioni terrestri in 2D
def generate_ce(m):
    angle = np.random.uniform(0, 2 * np.pi, m)
    Ce = np.vstack((R * np.cos(angle), R * np.sin(angle))).T
    return Ce

Ce = generate_ce(M)

# Genera una matrice binaria A che rappresenta l'associazione tra satelliti e stazioni terrestri
'''A = np.random.randint(0, 2, (N, M))
while np.sum(A, axis=1).min() == 0 or np.sum(A, axis=0).min() == 0:
    A = np.random.randint(0, 2, (N, M))
'''

A = np.zeros((N, M), dtype=int)
for i in range(N):
    j = np.random.randint(0, M)  # Scegli un indice casuale per la stazione
    A[i, j] = 1

print(A)




def objective_function(Cs):
    if Cs.ndim == 1:
        Cs = Cs.reshape(-1, 2)

    d = distance(Cs, Ce)
    f = 0
    # Terminologie di connessione e velocità
    for i in range(N):
        for j in range(M):
            term1 = w1 * (d[i, j] / V0)
            term2 = w2 * (1 - (0.5 / (1 + R / d[i, j]))) * A[i, j]
            f += (term1/0.15) + term2

    # Penalità normalizzate
    for i in range(N):
        for j in range(M):
            f += 1 * coordinate_constraints_violation(Cs) / (coord_limit_sat[1] - coord_limit_sat[0])
            f += 1 * min_max_distance_violation(d[i, j], dmax, dmin) / dmax
            f += 1 * one_to_one_relationship_violation(A)

    #penalties for satellites too close to each other
    for i in range(N):
        for j in range(N):
            if i != j:
                if np.abs(Cs[i, 0] - Cs[j, 0]) < 1000 and np.abs(Cs[i, 1] - Cs[j, 1]) < 1000:
                    f += 1e3
            

    return f





# Funzione per calcolare la distanza
def distance(Cs, Ce):
    # Assicurati che entrambi gli array siano bidimensionali
    if Cs.ndim == 1:
        Cs = Cs.reshape(1, -1)  # Trasforma in array 2D se è 1D
    if Ce.ndim == 1:
        Ce = Ce.reshape(1, -1)  # Trasforma in array 2D se è 1D

    # Cs e Ce devono avere la seconda dimensione pari a 2 per il contesto 2D
    if Cs.shape[1] != 2 or Ce.shape[1] != 2:
        raise ValueError("Both Cs and Ce must have exactly 2 columns representing coordinates.")

    # Utilizzo del broadcasting per calcolare le differenze
    Cs_expanded = Cs[:, np.newaxis, :]
    Ce_expanded = Ce[np.newaxis, :, :]
    diff = Cs_expanded - Ce_expanded
    d = np.sqrt(np.sum(diff**2, axis=2))
    return d




# Vincolo su distanza minima e massima
def min_max_distance_violation(d, dmax, dmin):
    if d > dmax:
        return d - dmax
    elif d < dmin:
        return dmin - d
    return 0


def one_to_one_relationship_violation(A):
    # Verifica che ogni satellite (riga) sia collegato a una sola stazione terrestre
    satellite_violations = sum(np.sum(A, axis=1) != 1)
    # Verifica che ogni stazione terrestre (colonna) sia collegata a un solo satellite
    ground_station_violations = sum(np.sum(A, axis=0) != 1)
    # La somma delle violazioni fornisce il totale delle non conformità alla relazione 1:1
    return satellite_violations + ground_station_violations


'''
def satellite_coverage_violation(A):
    return sum(np.sum(A, axis=1) == 0)

def ground_station_coverage_violation(A):
    return sum(np.sum(A, axis=0) == 0)
'''
# Vincolo su coordinate
def coordinate_constraints_violation(Cs):
    R = 6371  # raggio della Terra in km
    dmin = 2000  # distanza minima orbitale aggiuntiva in km
    dmax = 36000  # distanza massima orbitale aggiuntiva in km
    violation_total = 0

    min_orbital_limit = R + dmin
    max_orbital_limit = R + dmax

    for coord in Cs:
        for val in coord:  # Itera su ogni valore di x e y per ogni satellite
            if not (-max_orbital_limit <= val <= -min_orbital_limit or min_orbital_limit <= val <= max_orbital_limit):
                # Calcola la distanza massima al limite più vicino se fuori dai limiti
                if val > 0:  # Caso positivo
                    max_violation = max(abs(val - min_orbital_limit), abs(val - max_orbital_limit))
                else:  # Caso negativo
                    max_violation = max(abs(val + min_orbital_limit), abs(val + max_orbital_limit))

                violation_total += max_violation  # Aggiungi la violazione massima trovata

    return violation_total






# Auxiliar function for drawing ------------------------
def check_length_for_scheme(results, evaluation):
    if len(results) < evaluation:
        last_value = results[-1] if results else np.inf # i fill the results array to not raise an error due to different length
        results.extend([last_value] * (evaluation - len(results)))

    return results

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


# Parametri BOA
p = 0.2
a = 0.0001
[results, evaluations, ris_for_iteration, best_position] = BOA(objective_function, bound, num_butterflies=50, max_iter=1000, a=a, p=p)
[results2, evaluations2, ris_for_iteration2, best_position2] = BOA(objective_function, bound, num_butterflies=100, max_iter=1000, a=a, p=p)
[results3, evaluations3, ris_for_iteration3, best_position3] = BOA(objective_function, bound, num_butterflies=150, max_iter=1000, a=a, p=p)
#10846979.582646562


print('Iteration \t Best Fitness \t Num Evaluations \t Num Iterations')
print(f'First Run \t {ris_for_iteration[-1]:.3f} \t \t \t{evaluations} \t\t\t {len(ris_for_iteration)}')
print(f'Second Run \t {ris_for_iteration2[-1]:.3f} \t \t \t{evaluations2} \t\t\t {len(ris_for_iteration2)}')
print(f'Third Run \t {ris_for_iteration3[-1]:.3f} \t \t \t{evaluations3} \t\t\t {len(ris_for_iteration3)}')
print(best_position)
print(best_position2)
print(best_position3)

ris_eval_1 = check_length_for_scheme(results, evaluations)
ris_eval_2 = check_length_for_scheme(results2, evaluations2)
ris_eval_3 = check_length_for_scheme(results3, evaluations3)


evaluations = [evaluations, evaluations2, evaluations3]
results_eval = [ris_eval_1, ris_eval_2, ris_eval_3]
runs = ['First Run', 'Second Run', 'Third Run']
colors = ['red', 'green', 'blue']

draw_optimization_eval(evaluations, results_eval, runs, colors)

iterations = [len(ris_for_iteration), len(ris_for_iteration2), len(ris_for_iteration3)]
result_iter = [ris_for_iteration, ris_for_iteration2, ris_for_iteration3]


draw_optimization_iterations(iterations, result_iter, runs, colors)

# Predefined Optimization
#from scipy.optimize import minimize
#Cs = np.random.uniform(coord_limit_sat[0], coord_limit_sat[1], 2*N)
#res = minimize(objective_function, Cs, method='Nelder-Mead', options={'maxiter': 15000})
#print(res)


#Altro metodo di ottimizzazione
from scipy.optimize import differential_evolution
bounds = [coord_limit_sat] * 2*N
#result = differential_evolution(objective_function, bounds, maxiter=500)
#print(result)

print(A)

def draw_system(best_position, Ce, R, N, coord_limit_sat):
    fig, ax = plt.subplots()
    earth = plt.Circle((0, 0), R, color='blue', label='Earth')  # Earth as a circle at origin

    # Minimum and maximum orbit circles
    min_orbit = plt.Circle((0, 0), R + dmin, color='gray', linestyle='--', fill=False, linewidth=1.5)
    max_orbit = plt.Circle((0, 0), R + dmax, color='gray', linestyle='--', fill=False, linewidth=1.5)

    # Add Earth and orbits to plot
    ax.add_artist(earth)
    ax.add_artist(min_orbit)
    ax.add_artist(max_orbit)

    # Plot satellite positions and cones towards Earth
    for i in range(N):
        x, y = best_position[i*2], best_position[i*2+1]
        ax.plot(x, y, 'ro', label='Satellite' if i == 0 else "")  # Red points for satellites

    # Plot ground station positions
    for coord in Ce:
        ax.plot(coord[0], coord[1], 'go', label='Ground Station' if np.all(coord == Ce[0]) else "")  # Green points for ground stations

    # Set limits for the plot
    max_extent = R + dmax + 100  # adding a small buffer for visibility
    plt.xlim(-max_extent, max_extent)
    plt.ylim(-max_extent, max_extent)

    # Add grid, legend, and labels
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    plt.xlabel('x coordinate (km)')
    plt.ylabel('y coordinate (km)')
    plt.legend()
    plt.title('System Layout with Earth, Satellites, Ground Stations, and Orbits')

    plt.show()

draw_system(best_position, Ce, R, N, coord_limit_sat)
draw_system(best_position2, Ce, R, N, coord_limit_sat)
draw_system(best_position3, Ce, R, N, coord_limit_sat)

def calculate_average_latency(Cs):
    if Cs.ndim == 1:
        Cs = Cs.reshape(-1, 2)

    d = distance(Cs, Ce)
    total_latency = np.sum(d / V0)  # Somma delle latenze di tutti i satelliti
    average_latency = total_latency / len(Cs)  # Media delle latenze su tutti i satelliti
    return average_latency

def calculate_coverage(A, Cs, Ce):
    d = distance(Cs, Ce)
    total_coverage = np.sum((1 - (0.5 / (1 + R / d))) * A)
    total_possible_coverage = np.sum(A)
    coverage = total_coverage / total_possible_coverage
    return coverage


best_position = np.array( [ 12691.1027898  , -9353.69046016 , -8519.75781346 ,-11652.64997265,
                            12666.06062953 , 12666.06062953 , 10098.61824294, -12703.45252634,
                            -10054.85488291  , 8846.32339616] )
best_position2 = np.array([10970.94856073, 10855.52036757,  9367.37376852, -8821.18845187,
                           10855.52036757, -8821.18845187,  8821.18845187, 12318.02963614,
                           -8821.18845187, 10970.94856073])
best_position3 = np.array([  9586.01362612,   8558.60149261,  -8558.60149261, -11144.72959684,
                             -8558.60149261,  -8558.60149261,  -8558.60149261,  -9586.01362612,
                             -11144.72959684,  -8558.60149261]
                          )

average_latency_1 = calculate_average_latency(best_position)
coverage_1 = calculate_coverage(A, best_position.reshape(-1, 2), Ce)
# Calcola latenza media e copertura media per la soluzione ottimizzata 2
average_latency_2 = calculate_average_latency(best_position2)
coverage_2 = calculate_coverage(A, best_position2.reshape(-1, 2), Ce)

# Calcola latenza media e copertura media per la soluzione ottimizzata 3
average_latency_3 = calculate_average_latency(best_position3)
coverage_3 = calculate_coverage(A, best_position3.reshape(-1, 2), Ce)

# Stampa i risultati
print("Optimal solution 1")
print("Average Latency:", average_latency_1)
print("Copertura Media:", coverage_1)

print("Optimal solution 2")
print("Average Latency:", average_latency_2)
print("Copertura Media:", coverage_2)

print("Optimal solution 3")
print("Average Latency:", average_latency_3)
print("Copertura Media:", coverage_3)