import argparse
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as ct
from scipy.optimize import OptimizeWarning
from EDA.filter_data import filter_data
from EDA.normalize_data import normalize_X, normalize_y
from Feedforward_network.feedforward_network_load import feedforward_network_load
from GA.ga_model import ga
from GA.ga_evaluate import error_npred_ndesired

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned', category=np.RankWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module="deap.creator")
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")


# EDA : filter, normalize data
X_data_array_5000, y_data_array_5000=filter_data()
X_data_array_5000_normalized, X_data_array_5000_mean, X_data_array_5000_std=normalize_X(X_data_array_5000)
y_data_array_5000_normalized=normalize_y(y_data_array_5000)
frequencies=np.linspace(171309976000000, 222068487407407, 5000)


# LOAD FEEDFORWARD MODEL
device = torch.device('cpu')
feedforward_model=feedforward_network_load(device)

# GA
def main():
    parser = argparse.ArgumentParser(description="Calculate Best parameters")
    parser.add_argument('--n_desired', type=float, required=True, help='Value of effective index')
    parser.add_argument('--wavelength_desired', type=float, required=True, help='Value of wavelength in nanometers')
    args = parser.parse_args()

    f_desired= ct.c/(args.wavelength_desired*1e-9)

    best_param = ga(f_desired, args.n_desired, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    error, response_spectrum, _, n_pred, f_pred=error_npred_ndesired(best_param, args.n_desired, f_desired, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    best_param_denormalized=np.array(best_param)*X_data_array_5000_std[:3]+X_data_array_5000_mean[:3]
    print(f"\nBest w: {best_param_denormalized[0]}")
    print(f"Best DC: {best_param_denormalized[1]}")
    print(f"Best pitch: {best_param_denormalized[2]}\n")
    print(f"Error on n, predicted with those parameters : {error}\n")
    print(f"n desired : {args.n_desired}")
    print(f"n predicted with those parameters : {n_pred[0]}\n")
    print(f"f resonance predicted with those parameters : {f_pred[0]}")
    print(f'f_desired : {f_desired}')


    # Save results
    with open("results/result_param.txt", "a") as file:
        file.write(f"n={args.n_desired}, f={f_desired}, w={best_param_denormalized[0]}, DC={best_param_denormalized[1]}, pitch={best_param_denormalized[2]}\n")
    plt.plot(response_spectrum)
    plt.title(f"Predicted Response spectrum for w={best_param_denormalized[0]:.0f}, DC={best_param_denormalized[1]}:.2f, pitch={best_param_denormalized[2]:.0f}")
    plt.savefig(f"results/figures/response_spectrum_n_{args.n_desired}_f_{f_desired}.png")
    #plt.show()


if __name__ == "__main__":
    main()
