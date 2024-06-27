import random
from balayage_k import eval_n_eff_balayage_k
from ga_evaluate import eval_neff_params_frequency
from deap import base, creator, tools, algorithms

def ga(desired_frequency, desired_neff, feedforward_model, device, X_data_array_50_std, X_data_array_50_mean, filtered_frequencies):
    '''
    input : desired_frequency, desired_neff
    output : best_ind=[w,DC, pitch]
    Renvoie les meilleurs paramètres w,DC, pitch tels que y(f_desired,w,DC, pitch)=n_pred proche de desired_neff
    '''
    print("Running genetic algorithm...")
    # Création des types de base
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_w", random.uniform,-1.9,1.57)
    toolbox.register("attr_dc", random.uniform, -2.11, 1.64)
    toolbox.register("attr_pitch", random.uniform, -2.8,  1.31)

    # Initialiser les individus et la population
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_w, toolbox.attr_dc, toolbox.attr_pitch), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Définir la fonction de fitness
    def evalNeff(individual):
        w, DC, pitch = individual
        neff = eval_n_eff_balayage_k([w, DC, pitch], desired_frequency, feedforward_model, device, X_data_array_50_std, X_data_array_50_mean, filtered_frequencies)
        n_pred, response_spectrum = eval_neff_params_frequency([w, DC, pitch], desired_neff, desired_frequency, feedforward_model, device, X_data_array_50_std, X_data_array_50_mean, filtered_frequencies)
        if len(n_pred[0])==0:
            neff*=10
        return (abs(neff - desired_neff)),

    toolbox.register("evaluate", evalNeff)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Paramètres de l'algorithme génétique
    population = toolbox.population(n=300)
    ngen = 30
    cxpb = 0.5
    mutpb = 0.2

    # Exécuter l'algorithme génétique
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)
    
    # Afficher les meilleurs résultats
    best_ind = tools.selBest(population, 1)[0]

    return(best_ind)