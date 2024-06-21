import random
import numpy as np
from deap import base, creator, tools, algorithms
import torch
import torch.nn as nn
from ../../Feedforward_network/feedforward_network_model import FeedForwardNN

def genetic_algorithm(desired_output_GA):
    # Paramètres du réseau
    input_size_ffn = 4
    output_size_ffn = 50
    best_hidden_sizes_ffn = [300, 300, 300, 300]
    # Entraîner le modèle avec les meilleurs paramètres
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)                        
    feedforward_model.load_state_dict(torch.load('../../Feedforward_network/feedforward_model_trained.pth'))
    feedforward_model.eval()

    # Définir la fonction de perte
    criterion = nn.MSELoss()

    desired_output = desired_output_GA

    def evaluate_individual(individual):
        w, DC, pitch, k = individual
        # Convertir les paramètres optimisés en entrée pour le modèle
        X_opt = np.array([[w, DC, pitch, k]])
        X_opt = torch.tensor(X_opt, dtype=torch.float32)

        # Prédire les valeurs du champ électrique
        with torch.no_grad():
            predicted_field = feedforward_model(X_opt)

        # Calculer la perte
        loss = criterion(predicted_field, desired_output_GA).item()
        return loss,

    # Supprimer les classes existantes dans creator pour éviter les avertissements
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    # Définir le type de problème et la structure de l'individu
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Définir les opérateurs génétiques
    toolbox = base.Toolbox()

    param_bounds = {
        "w": (-1.9, 1.57),
        "DC": (-2.11, 1.64),
        "pitch": (-2.8, 1.31),
        "k": (-2.6, 3.34)
    }


    # Enregistrer la création de l'individu et de la population dans le Toolbox
    toolbox.register("attr_w", np.random.uniform, param_bounds["w"][0], param_bounds["w"][1])
    toolbox.register("attr_DC", np.random.uniform, param_bounds["DC"][0], param_bounds["DC"][1])
    toolbox.register("attr_pitch", np.random.uniform, param_bounds["pitch"][0], param_bounds["pitch"][1])
    toolbox.register("attr_k", np.random.uniform, param_bounds["k"][0], param_bounds["k"][1])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_w, toolbox.attr_DC, toolbox.attr_pitch, toolbox.attr_k), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mutate_corrected(individual, indpb):
        param_names = ["w","DC","pitch","k"]
        for i in range(len(individual)):
            if random.random() < indpb:
                param_name = param_names[i]
                lower_bound, upper_bound = param_bounds[param_name]

                # Vérifier et corriger si nécessaire
                if individual[i] < lower_bound:
                    individual[i] = lower_bound
                elif individual[i] > upper_bound:
                    individual[i] = upper_bound

        return individual,

    # Appliquer la correction des bornes à l'opérateur mutate
    toolbox.register("mutate", mutate_corrected, indpb=0.05)

    # Définir l'opérateur de croisement
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # Définir la fonction d'évaluation
    toolbox.register("evaluate", evaluate_individual)

    # Sélectionner la fonction de sélection
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=200)  # Créer une population initiale avec 50 individus
    ngen = 40
    cxpb = 0.5
    mutpb = 0.2

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        top10 = tools.selBest(population, k=10)
        #print(f"Gen: {gen}, Best Loss: {top10[0].fitness.values[0]}")

    return tools.selBest(population, k=1)