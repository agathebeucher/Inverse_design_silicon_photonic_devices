import argparse
import torch
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as ct
from scipy.optimize import OptimizeWarning
from EDA.load_data import load_data, create_datasets
from EDA.normalize_data import normalize_X, normalize_y
from Feedforward_network.feedforward_network_load import feedforward_network_load
from GA.ga_model import ga
from GA.ga_evaluate import error_npred_ndesired

warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned', category=np.RankWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module="deap.creator")
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Covariance of the parameters could not be estimated")


# EDA : load, normalize data
# Load and normalize data
X_data_array_5000, y_data_array_5000=load_data()
X_data_array_5000_normalized, X_data_array_5000_mean, X_data_array_5000_std=normalize_X(X_data_array_5000)
y_data_array_5000_normalized=normalize_y(y_data_array_5000)

# Create Training/Test/Validation datasets
test_size=0.2 # 80% training, 10% validation, 10% test
X_train_5000_normalized, y_train_5000_normalized, X_test_5000_normalized, y_test_5000_normalized, X_val_5000_normalized, y_val_5000_normalized=create_datasets(X_data_array_5000_normalized, y_data_array_5000_normalized, test_size)
frequencies=np.linspace(171309976000000, 222068487407407, 5000)

# LOAD FEEDFORWARD MODEL (pre-trained)
device = torch.device('cpu')
feedforward_model=feedforward_network_load(device)

# GA
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate Best parameters")
    parser.add_argument('--n_desired', type=float, required=True, help='Value of effective index')
    parser.add_argument('--wavelength_desired', type=float, required=True, help='Value of wavelength in nanometers')
    parser.add_argument('--fixed_w', type=float, help='Fixed value for w')
    parser.add_argument('--fixed_DC', type=float, help='Fixed value for DC')
    parser.add_argument('--fixed_pitch', type=float, help='Fixed value for pitch')
    args = parser.parse_args()

    # Convert the desired wavelength to frequency
    f_desired = ct.c / (args.wavelength_desired * 1e-9)

    # Store the fixed parameters in a dictionary
    fixed_params = {
        'w': args.fixed_w,
        'DC': args.fixed_DC,
        'pitch': args.fixed_pitch
    }

    # Normalize the fixed parameters if they are provided
    if fixed_params['w'] is not None:
        fixed_params['w'] = (fixed_params['w']- X_data_array_5000_mean[0])/ X_data_array_5000_std[0]
    if fixed_params['DC'] is not None:
        fixed_params['DC'] = (fixed_params['DC']- X_data_array_5000_mean[1])/ X_data_array_5000_std[1]
    if fixed_params['pitch'] is not None:
        fixed_params['pitch'] = (fixed_params['pitch']- X_data_array_5000_mean[2])/ X_data_array_5000_std[2]

    # Run the genetic algorithm to find the best parameters
    best_param = ga(f_desired, args.n_desired, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies, fixed_params)
    # Evaluate the error and get the predicted response spectrum
    error, response_spectrum, _, n_pred, f_pred = error_npred_ndesired(best_param, args.n_desired, f_desired, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    # Denormalize the best parameters
    best_param_denormalized = np.array(best_param) * X_data_array_5000_std[:3] + X_data_array_5000_mean[:3]
    
    # Print the best parameters and the corresponding error
    print(f"\nBest w: {best_param_denormalized[0]}")
    print(f"Best DC: {best_param_denormalized[1]}")
    print(f"Best pitch: {best_param_denormalized[2]}\n")
    print(f"Error on n, predicted with those parameters : {error}\n")
    print(f"n desired : {args.n_desired}")
    print(f"n predicted with those parameters : {n_pred[0]}\n")
    print(f"f resonance predicted with those parameters : {f_pred[0]}")
    print(f'f_desired : {f_desired}')

    # Save the results to a file
    with open("results/result_param.txt", "a") as file:
        file.write(f"n={args.n_desired}, f={f_desired}, w={best_param_denormalized[0]}, DC={best_param_denormalized[1]}, pitch={best_param_denormalized[2]}\n")
    
    # Plot and save the predicted response spectrum
    plt.plot(response_spectrum)
    plt.title(f"Predicted Response spectrum for w={best_param_denormalized[0]:.0f}, DC={best_param_denormalized[1]:.2f}, pitch={best_param_denormalized[2]:.0f}")
    plt.savefig(f"results/figures/response_spectrum_n_{args.n_desired}_f_{f_desired}.png")

# Entry point of the script
if __name__ == "__main__":
    main()
