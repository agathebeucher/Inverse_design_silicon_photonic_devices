import random
import torch
import numpy as np
import sys
import time
import threading
import scipy.constants as ct
from .balayage_k import eval_n_eff_balayage_k
from .balayage_k import n_eff_one
from deap import base, creator, tools, algorithms

def eval_neff_params_frequency(best_params, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Evaluates the effective index predicted by the feedforward network with the best parameters w, DC, pitch, k,
    and the k derived from the desired n and the chosen frequency.
    
    Input:
    - best_params: The best parameters [w, DC, pitch]
    - desired_neff: The desired effective index
    - desired_frequency: The desired frequency
    - feedforward_model: The feedforward neural network model
    - device: The device to run the model (CPU/GPU)
    - X_data_array_5000_std: Standard deviations for normalization
    - X_data_array_5000_mean: Means for normalization
    - frequencies: The frequency spectrum
    
    Output:
    - n_response: The predicted effective index and frequency
    - response_np: The predicted spectrum
    '''
    # Calculate the desired k value
    k_desired = desired_neff * (2 * np.pi) * desired_frequency / ct.c
    # Normalize the k value
    k_desired_normalized = (k_desired - X_data_array_5000_mean[3]) / X_data_array_5000_std[3]
    # Append the normalized k value to the best parameters
    best_params_plus_k = best_params + [k_desired_normalized]
    # Convert the parameters to a tensor
    best_params_tensor = torch.tensor(np.array(best_params_plus_k), dtype=torch.float32).to(device)
    # Predict the spectrum using the feedforward model
    response_np = feedforward_model(best_params_tensor).to(device).detach().cpu().numpy()   
    # Denormalize the parameters
    best_params_denormalized = np.array(best_params_plus_k) * X_data_array_5000_std + X_data_array_5000_mean
    # Calculate the effective index and frequency
    n_response = n_eff_one(best_params_denormalized, response_np, frequencies)
    return n_response, response_np


# Bounds for the parameters w, DC, and pitch
BOUNDS = {
    'w': (-1.9, 1.57),
    'DC': (-2.11, 1.64),
    'pitch': (-2.8, 1.31)
}

def check_bounds(individual):
    ''' 
    Check and correct the parameters if they are out of bounds.
    Input:
    - individual: The list of parameters [w, DC, pitch]
    Output:
    - individual: The corrected list of parameters
    '''
    for i, (param, (lower, upper)) in enumerate(BOUNDS.items()):
        if individual[i] < lower:
            individual[i] = lower
        elif individual[i] > upper:
            individual[i] = upper
    return individual

def ga(desired_frequency, desired_neff, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies, fixed_params):
    '''
    Runs a genetic algorithm to find the best parameters [w, DC, pitch] such that y(f_desired, w, DC, pitch) = n_pred is close to desired_neff.
    
    Input:
    - desired_frequency: The target frequency
    - desired_neff: The target effective index
    - feedforward_model: The feedforward neural network model
    - device: The device to run the model (CPU/GPU)
    - X_data_array_5000_std: Standard deviations for normalization
    - X_data_array_5000_mean: Means for normalization
    - frequencies: The frequency spectrum
    - fixed_params: Dictionary of fixed parameters { 'w': value, 'DC': value, 'pitch': value }
    
    Output:
    - best_ind: The best parameters [w, DC, pitch]
    '''
    print("Running genetic algorithm...")
    start = time.time()

    # Create the base types for the genetic algorithm
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Register the attribute generation functions
    if fixed_params['w'] is None:
        toolbox.register("attr_w", random.uniform, *BOUNDS['w'])
    if fixed_params['DC'] is None:
        toolbox.register("attr_dc", random.uniform, *BOUNDS['DC'])
    if fixed_params['pitch'] is None:
        toolbox.register("attr_pitch", random.uniform, *BOUNDS['pitch'])

    def init_individual(container):
        individual = container()
        if fixed_params['w'] is not None:
            individual.append(fixed_params['w'])
        else:
            individual.append(toolbox.attr_w())

        if fixed_params['DC'] is not None:
            individual.append(fixed_params['DC'])
        else:
            individual.append(toolbox.attr_dc())

        if fixed_params['pitch'] is not None:
            individual.append(fixed_params['pitch'])
        else:
            individual.append(toolbox.attr_pitch())
        return individual

    # Initialize individuals and the population
    toolbox.register("individual", init_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Define the fitness function
    def evalNeff(individual):
        individual = check_bounds(individual)
        w, DC, pitch = individual
        neff = eval_n_eff_balayage_k([w, DC, pitch], desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
        if neff == 0:
            return (abs(neff - desired_neff),)
        else:
            n_pred, response_spectrum = eval_neff_params_frequency([w, DC, pitch], desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
            if len(n_pred[0]) == 0:
                diff_f = 10
            else:
                diff_f = abs(n_pred[0][0] - neff)
            return (abs(neff - desired_neff) + diff_f,)

    toolbox.register("evaluate", evalNeff)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Genetic algorithm parameters
    population = toolbox.population(n=300)
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2
    min_loss = 1e-3

    for gen in range(ngen):
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen=1, verbose=1)

        # Display the best individuals after each generation
        best_ind = tools.selBest(population, 1)[0]
        best_fitness = best_ind.fitness.values[0]

        # Check the stopping criterion
        if best_fitness < min_loss:
            break

    # Display the best results
    best_ind = tools.selBest(population, 1)[0]
    end = time.time()
    elapsed_time = end - start
    print(f'Elapsed time: {elapsed_time} seconds')
    return best_ind