import argparse
import torch
import matplotlib.pyplot as plt
from scipy import constants as ct
from EDA.filter_data import filter_data
from EDA.normalize_data import normalize_X, normalize_y
from Feedforward_network.feedforward_network_model import FeedForwardNN
from GA.ga_model import ga
from GA.ga_evaluate import error_npred_ndesired

# EDA
X_data_array_50, y_data_array_50=filter_data()
X_data_array_50_normalized=normalize_X(X_data_array_50)
y_data_array_50_normalized=normalize_y(y_data_array_50)

# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FEEDFORWARD MODEL
input_size_ffn = 4
output_size_ffn = 50
best_hidden_sizes_ffn = [300, 300, 300, 300]
feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)                        
feedforward_model.load_state_dict(torch.load('./Feedforward_network/feedforward_model_trained.pth'))
feedforward_model.eval()

# GA
def main():
    parser = argparse.ArgumentParser(description="Calculate Best parameters")
    parser.add_argument('--n_desired', type=float, required=True, help='Value of effective index')
    parser.add_argument('--wavelength_desired', type=float, required=True, help='Value of wavelength in nanometers')
    args = parser.parse_args()

    f_desired= ct.c/(args.wavelength_desired*1e-9)

    best_param = ga(args.f_desired, args.n_desired)
    error, response_spectrum, _, n_pred, f_pred=error_npred_ndesired(best_param, args.n_desired, f_desired)

    print(f"Best w: {best_param[0]}")
    print(f"Best DC: {best_param[1]}")
    print(f"Best pitch: {best_param[2]}")
    print(f"Error predicted with those parameters : {error}")
    print(f"n predicted with those parameters : {n_pred}")
    print(f"f resonance predicted with those parameters : {f_pred}")

    plt.plot(response_spectrum)
    plt.title("Predicted Response spectrum with best parameters")
    plt.show()
    

if __name__ == "__main__":
    main()
