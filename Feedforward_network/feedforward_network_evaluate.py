import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def feedforward_network_evaluate(feedforward_model, X_test, y_test):
    # Mettre le modèle en mode d'évaluation
    feedforward_model.eval()

    # Effectuer des prédictions sur les données de test
    with torch.no_grad():
        y_pred_tensor = feedforward_model(X_test)

    # Convertir les prédictions en un tableau numpy
    y_pred_array = y_pred_tensor.numpy()
    y_test_array = y_test.numpy()
    X_test_array = X_test.numpy()

    def weighted_error(true_values, predicted_values, sigma=0.5):
        peak_index = np.argmax(true_values)
        num_values = len(true_values)
        weights = np.exp(-((np.arange(num_values) - peak_index) ** 2) / (2 * sigma ** 2))
        error = np.sum(weights * (true_values - predicted_values) ** 2) / num_values
        return error

    def calculate_error_percentages(true_data, predicted_data, thresholds):
        # Calculer les erreurs pour chaque cas de test
        errors = [weighted_error(true, pred) for true, pred in zip(true_data, predicted_data)]
        # Initialiser les comptages
        counts = {threshold: 0 for threshold in thresholds}
        # Compter le nombre de données pour chaque seuil d'erreur
        for error in errors:
            for threshold in thresholds:
                if error <= threshold:
                    counts[threshold] += 1
        # Calculer les pourcentages
        total_count = len(errors)
        percentages = {threshold: (count / total_count) * 100 for threshold, count in counts.items()}
        # Combiner les nombres et les pourcentages dans un seul dictionnaire
        results = {threshold: {'count': counts[threshold], 'percentage': percentages[threshold]} for threshold in thresholds}
        return results

    # Supposons que y_pred_normalized et y_test_50_normalized soient déjà définis
    thresholds = [0.001]
    percentages = calculate_error_percentages(y_test_array, y_pred_array, thresholds)

    print("Percentages and counts of data with specific errors:")
    for threshold, info in percentages.items():
        print(f"Error <= {threshold}: {info['count']} data points, {info['percentage']:.2f}%")

    def plot_original_and_normalized_separate(data_pred, data_test, base_number=0, n_plt=8, m_plt=8):
        plt.rcParams["figure.figsize"]= 20,20
        fig,axs = plt.subplots(n_plt,m_plt)
        for i in range(0,n_plt):
            for j in range(0,m_plt):        
                test_case = i*n_plt + j + base_number
                x = np.linspace(0, 1, 50)
                axs[i,j].plot(data_test[test_case], label='Valeurs réelles normalisé')
                axs[i,j].plot(data_pred[test_case], label='Prédictions')    
        plt.savefig('Result_testdataset_feedforward_model.png')
        #plt.show()

    return 0;