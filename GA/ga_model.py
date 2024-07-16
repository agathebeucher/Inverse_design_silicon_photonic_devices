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
    Evalue l'indice effectif prédit par le feed_forward network avec les meilleurs paramètres w,DC,pitch,k, 
    et le k déduit du n voulu et de la fréquence choisie
    input : w,DC, pitch, f_desired, n_desired
    output : n_pred, f_pred, spectrum_pred
    '''
    k_desired=desired_neff*(2*np.pi)*desired_frequency/ct.c
    k_desired_normalized=(k_desired-X_data_array_5000_mean[3])/X_data_array_5000_std[3]
    best_params_plus_k=best_params+[k_desired_normalized]
    best_params_tensor = torch.tensor(np.array(best_params_plus_k), dtype=torch.float32).to(device)
    response_np=feedforward_model(best_params_tensor).to(device).detach().cpu().numpy()   
    best_params_denormalized=np.array(best_params_plus_k)*X_data_array_5000_std+X_data_array_5000_mean
    n_response=n_eff_one(best_params_denormalized, response_np, frequencies)
    return(n_response, response_np)

BOUNDS = {
    'w': (-1.9, 1.57),
    'DC': (-2.11, 1.64),
    'pitch': (-2.8, 1.31)
}

def check_bounds(individual):
    ''' Check and correct the param if they are out of bounds'''
    for i,(param, (lower, upper)) in enumerate(BOUNDS.items()):
        if individual[i] < lower:
            individual[i] = lower
        elif individual[i] > upper:
            individual[i] = upper
    return individual

def ga(desired_frequency, desired_neff, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    input : desired_frequency, desired_neff
    output : best_ind=[w,DC, pitch]
    Renvoie les meilleurs paramètres w,DC, pitch tels que y(f_desired,w,DC, pitch)=n_pred proche de desired_neff
    '''
    print("Running genetic algorithm...")    
    start=time.time()
    '''
    # Thread pour afficher le temps écoulé
    global stop_flag
    stop_flag = False
    timer_thread = threading.Thread(target=display_elapsed_time)
    timer_thread.start()
    '''
    # Création des types de base
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_w", random.uniform,*BOUNDS['w'])
    toolbox.register("attr_dc", random.uniform, *BOUNDS['DC'])
    toolbox.register("attr_pitch", random.uniform, *BOUNDS['pitch'])

    # Initialiser les individus et la population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_w, toolbox.attr_dc, toolbox.attr_pitch), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Définir la fonction de fitness
    def evalNeff(individual):
        individual=check_bounds(individual)
        w, DC, pitch = individual
        neff = eval_n_eff_balayage_k([w, DC, pitch], desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
        if neff==0:
            #print(f'neff: {neff}')
            return(abs(neff - desired_neff),)
        else :
            n_pred, response_spectrum = eval_neff_params_frequency([w, DC, pitch], desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
            if len(n_pred[0])==0:
                diff_f=10
                #print(f'neff: {neff}, diff_f: {diff_f}')
            else:
                diff_f=abs(n_pred[0][0]-neff)
                #print(f'neff: {neff}, n_pred: {n_pred[0][0]}, diff_f: {diff_f}')
            return (abs(neff - desired_neff)+diff_f,)

    toolbox.register("evaluate", evalNeff)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Paramètres de l'algorithme génétique
    population = toolbox.population(n=300)
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2
    min_loss = 1e-3

    for gen in range(ngen):
        algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen=1, verbose=1)

        # Afficher les meilleurs individus après chaque génération
        best_ind=tools.selBest(population, 1)[0]
        best_fitness=best_ind.fitness.values[0]

        # Vérifier le critère d'arrêt
        if best_fitness < min_loss:
            break
    
    # Arrêter le thread du timer
    #stop_flag = True
    #timer_thread.join()

    # Afficher les meilleurs résultats
    best_ind = tools.selBest(population, 1)[0]
    end=time.time()
    elapsed_time=end-start
    print(f'Elapsed time: {elapsed_time} seconds')
    return(best_ind)

def display_elapsed_time(total_time=1000):
    start_time = time.time()
    while not stop_flag:
        elapsed_time = int(time.time() - start_time)
        progress = elapsed_time / total_time
        bar_length = 50
        block = int(round(bar_length * progress))
        bar = "#" * block + "-" * (bar_length - block)
        sys.stdout.write(f'\rRunning genetic algorithm : [{bar}] {elapsed_time}/{total_time} seconds')
        sys.stdout.flush()
        time.sleep(1)
        if elapsed_time >= total_time:
            break
    elapsed_time = int(time.time() - start_time)
    progress = elapsed_time / total_time
    block = int(round(bar_length * progress))
    bar = "#" * block + "-" * (bar_length - block)
    sys.stdout.write(f'\rRunning genetic algorithm : [{bar}] {elapsed_time}/{total_time} seconds\n')
    sys.stdout.flush()