import argparse
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as ct
from EDA.filter_data import filter_data
from EDA.normalize_data import normalize_X, normalize_y
from Feedforward_network.feedforward_network_model import FeedForwardNN
from GA.ga_model import ga
from GA.ga_evaluate import error_npred_ndesired

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned', category=np.RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")

# EDA
X_data_array_50, y_data_array_50=filter_data()
X_data_array_50_normalized, X_data_array_50_mean, X_data_array_50_std=normalize_X(X_data_array_50)
y_data_array_50_normalized=normalize_y(y_data_array_50)
filtered_frequencies=np.linspace(171309976000000, 222068487407407, 50)


# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FEEDFORWARD MODEL
input_size_ffn = 4
output_size_ffn = 50
best_hidden_sizes_ffn = [955, 925, 1005, 407, 580, 1309]
feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)
if torch.cuda.is_available():
    checkpoint = torch.load('./Feedforward_network/feedforward_model_trained_gpu.pth', map_location=torch.device('cpu'))
else:
    checkpoint = torch.load('./Feedforward_network/feedforward_model_trained_cpu.pth', map_location=torch.device('cpu'))
feedforward_model.load_state_dict(checkpoint)
feedforward_model.eval()

# GA
def main():
    parser = argparse.ArgumentParser(description="Calculate Best parameters")
    parser.add_argument('--n_desired', type=float, required=True, help='Value of effective index')
    parser.add_argument('--wavelength_desired', type=float, required=True, help='Value of wavelength in nanometers')
    args = parser.parse_args()

    f_desired= ct.c/(args.wavelength_desired*1e-9)

    best_param = ga(f_desired, args.n_desired, feedforward_model, device, X_data_array_50_std, X_data_array_50_mean, filtered_frequencies)
    error, response_spectrum, _, n_pred, f_pred=error_npred_ndesired(best_param, args.n_desired, f_desired, feedforward_model, device, X_data_array_50_std, X_data_array_50_mean, filtered_frequencies)
    best_param_denormalized=np.array(best_param)*X_data_array_50_std[:3]+X_data_array_50_mean[:3]
    print(f"\nBest w: {best_param_denormalized[0]}")
    print(f"Best DC: {best_param_denormalized[1]}")
    print(f"Best pitch: {best_param_denormalized[2]}\n")
    print(f"Error on n, predicted with those parameters : {error}\n")
    print(f"n desired : {args.n_desired}")
    print(f"n predicted with those parameters : {n_pred[0]}\n")
    print(f"f resonance predicted with those parameters : {f_pred[0]}")
    print(f'f_desired : {f_desired}')

    plt.plot(response_spectrum)
    plt.title("Predicted Response spectrum with best parameters")
    #plt.show()
    

if __name__ == "__main__":
    main()
