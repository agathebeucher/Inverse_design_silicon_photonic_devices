import torch
import matplotlib.pyplot as plt
import numpy as np
from ./ga_model import genetic_algorithm
from ../../Feedforward_network/feedforward_network_model import FeedForwardNN

def ga_evaluate(y_test):
    # Paramètres du réseau
    input_size_ffn = 4
    output_size_ffn = 50
    best_hidden_sizes_ffn = [300, 300, 300, 300]
    # Entraîner le modèle avec les meilleurs paramètres
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)                        
    feedforward_model.load_state_dict(torch.load('../../Feedforward_network/feedforward_model_trained.pth'))
    feedforward_model.eval()
    
    best_params_pred=[]
    predicted_fields=[]
    for i in range(len(y_test)):
        desired_output=y_test[i]
        best_params=genetic_algorithm(desired_output)
        predicted_field=feedforward_model(torch.tensor(best_params, dtype=torch.float32))
        best_params_pred.append(best_params)
        predicted_fields.append(predicted_field)

    return(best_params_pred,predicted_fields)

def ga_evaluate_spectre(best_params_pred, predicted_fields, y_test):
    # Paramètres du réseau
    input_size_ffn = 4
    output_size_ffn = 50
    best_hidden_sizes_ffn = [300, 300, 300, 300]
    # Entraîner le modèle avec les meilleurs paramètres
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)                        
    feedforward_model.load_state_dict(torch.load('../../Feedforward_network/feedforward_model_trained.pth'))
    feedforward_model.eval()
    
    plt.rcParams["figure.figsize"] = 20, 20
    fig, axs = plt.subplots(5, 5)
    
    # Tracer les valeurs réelles et prédites selon les indices sélectionnés
    for idx, (i, j) in enumerate(np.ndindex(5, 5)):
        test_case = i*j 
        predicted_field=feedforward_model(torch.tensor(best_params_pred[test_case], dtype=torch.float32)).detach().numpy()
        axs[i, j].plot(y_test[test_case], label='Valeurs réelles normalisées')
        axs[i, j].plot(predicted_fields[test_case], label='Prédictions')

    plt.tight_layout()
    plt.savefig('Result_testdataset_GA_model.png')
    #plt.show()

def ga_evaluate_param(best_params_pred, predicted_fields, y_test, X_data_array_50_std, X_data_array_50_mean, X_test_array):
    best_params_pred_flattened = [item[0] for item in best_params_pred]*X_data_array_50_std+X_data_array_50_mean
    for i in range(5):
        print(f"Réel : {best_params_pred_flattened[i]}")
        print(f"Prédit : {X_test_array[i]}")