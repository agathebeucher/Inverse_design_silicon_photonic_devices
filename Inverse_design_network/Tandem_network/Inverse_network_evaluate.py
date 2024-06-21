import torch
import matplotlib.pyplot as plt
import numpy as np
from ../../Feedforward_network/feedforward_network_model import FeedForwardNN
from .Inverse_network_model import InverseNetwork

def Inverse_network_evaluate(X_test_inverse, y_test_inverse):
    # Paramètres du réseau
    input_size_ffn = 4
    output_size_ffn = 50
    best_hidden_sizes_ffn = [300, 300, 300, 300]
    # Entraîner le modèle avec les meilleurs paramètres
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)                        
    feedforward_model.load_state_dict(torch.load('../../Feedforward_network/feedforward_model_trained.pth'))
    feedforward_model.eval()

    # Paramètres du réseau
    # Entraîner le modèle inverse avec les meilleurs hyperparamètres trouvés
    best_hidden_sizes_inverse = [300, 300,300,300,300]
    input_size_inverse=50
    output_size_inverse=4
    # Entraîner le modèle avec les meilleurs paramètres
    inverse_model = InverseNetwork(input_size_inverse, best_hidden_sizes_inverse, output_size_inverse)                     
    inverse_model.load_state_dict(torch.load('./inverse_model_trained.pth'))
    inverse_model.eval()

    # Convertir les tenseurs en arrays numpy pour une manipulation plus facile
    predicted_design_params_test=inverse_model(X_test_inverse)
    predicted_field_test=feedforward_model(predicted_design_params_test)

    predicted_design_params_test_np=predicted_design_params_test.detach().numpy()
    predicted_field_test_np = predicted_field_test.detach().numpy()
    X_test_inverse_np = X_test_inverse.numpy()

    def plot_original_and_normalized_separate(data_pred, data_test, base_number=0, n_plt=4, m_plt=5):
        plt.rcParams["figure.figsize"]= 20,20
        fig, axs = plt.subplots(n_plt, m_plt)

        for i in range(n_plt):
            for j in range(m_plt):        
                test_case = i * m_plt + j + base_number
                x = np.linspace(0, 1, 50)
                axs[i,j].plot(data_test[test_case], label='Valeurs réelles normalisées')
                axs[i,j].plot(data_pred[test_case], label='Prédictions')    
                axs[i,j].set_title(f'Exemple {test_case + 1}')
                axs[i,j].legend()
        
        plt.savefig('Result_testdataset_feedforward_model.png')
        #plt.show()

    # Utilisation de la fonction avec vos données
    plot_original_and_normalized_separate(predicted_field_test_np, X_test_inverse_np, n_plt=4, m_plt=5)