import numpy as np
from scipy.signal import argrelextrema
from scipy import constants as ct
from .approx_gauss import approx_gauss
import torch

def conversion_k(k_usi, grating_pitch):
    '''
    Function to convert k_usi to a different unit based on grating pitch
    '''
    return 2 * np.pi * k_usi / grating_pitch

def find_1D_localmax_one(data, threshold, sample_dist):
    '''
    Function to find local maxima in a 1D dataset that exceed a specified threshold
    '''
    idx = argrelextrema(data=data, comparator=np.greater, order=sample_dist)[0]
    idx = idx[np.where(data[idx] > threshold)]
    max_value = max(data)
    maxima_amplitude = data[idx]
    maxima_amplitude_sorted = np.sort(maxima_amplitude)
    maxima_amplitude_sorted = np.flipud(maxima_amplitude_sorted)
    c = np.empty(maxima_amplitude_sorted.size, dtype=np.int64)
    for i in range(maxima_amplitude_sorted.size):
        temp = maxima_amplitude.tolist().index(maxima_amplitude_sorted[i])
        c[i] = idx[temp]
    if len(c) > 0:
        if max_value - data[c[0]] > 0.5:
            return []
        else:
            return [c[0]]
    else:
        return []


def n_eff_one(X_data, y_data, freq):
    '''
    Function to calculate the effective index (n_eff) for a given X_data and y_data over a range of frequencies
    '''
    # Analysis parameters
    peaks_threshold = 0.01
    sample_dist = 10

    # Load data
    freq_axis = freq
    grating_pitch = X_data[2] * 1e-9
    k_value = X_data[3]
    fs_all = y_data

    # Find peaks in the spectrum
    peaks_idx = find_1D_localmax_one(fs_all, peaks_threshold, sample_dist)
    if len(peaks_idx) == 1:
        f_band = freq_axis[peaks_idx]
        bloch_index = k_value * 3e8 / (2 * np.pi * f_band)
        return [bloch_index[0]], [f_band[0]]
    else:
        return [], []


def calculate_func_neff_f(X_params_without_k_normalized, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Function to calculate the effectif indexes and corresponding resonance frequencies for different values of k at 
    a given set of parameters using a feedforward neural network model
    Input: X_params_without_k_normalized = [w, DC, pitch]
    Output: (n_k_list, f_k_list) -> points of the n = y(f) curve
    '''
    grating_pitch_denormalized = X_params_without_k_normalized[2] * X_data_array_5000_std[2] + X_data_array_5000_mean[2]
    k_list_denormalized = [conversion_k(round(num, 2), grating_pitch_denormalized * 1e-9) for num in np.arange(0.25, 0.50, 0.01)]
    k_list_normalized = [(k_denormalized - X_data_array_5000_mean[3]) / X_data_array_5000_std[3] for k_denormalized in k_list_denormalized]
    X_params_list = np.array([X_params_without_k_normalized + [k] for k in k_list_normalized])
    
    spectrum_k_list = []
    n_k_list = []
    f_k_list = []
    n_k_list_temp = []
    f_k_list_temp = []

    # Predict frequency spectrum for each k
    for i in range(len(k_list_normalized)):
        X_params_tensor = torch.tensor(np.array(X_params_list[i]), dtype=torch.float32).to(device)
        spectrum_k = feedforward_model(X_params_tensor).to(device)
        spectrum_k_list.append(spectrum_k.detach().cpu().numpy())

    # Predict resonance frequencies and effective indices for each k
    for i in range(len(spectrum_k_list)):
        X_params_denormalized = X_params_list[i] * X_data_array_5000_std + X_data_array_5000_mean
        n_eff_k, f_k = n_eff_one(X_params_denormalized, spectrum_k_list[i], frequencies)
        n_k_list_temp.append(n_eff_k)
        f_k_list_temp.append(f_k)

    for i in range(len(n_k_list_temp)):
        if n_k_list_temp[i] != []:
            mse, y_fit = approx_gauss(spectrum_k_list[i])
            if i != 0 and i != len(n_k_list_temp) - 1:
                if (n_k_list_temp[i + 1] != [] or n_k_list_temp[i - 1] != []) and y_fit is not None:
                    if mse < 0.0001:
                        n_k_list.append(n_k_list_temp[i][0])
                        f_k_list.append(f_k_list_temp[i][0])
            else:
                if y_fit is not None:
                    if mse < 0.0001:
                        n_k_list.append(n_k_list_temp[i][0])
                        f_k_list.append(f_k_list_temp[i][0])
    return n_k_list, f_k_list


def determine_initial_trend(values, lookahead=5):
    '''
    Function to determine the initial trend in a series of values
    '''
    if len(values) < 2:
        return "increasing"
    filtered_values = [values[0]]
    for value in values[1:]:
        if value != filtered_values[-1]:
            filtered_values.append(value)
    initial_values = filtered_values[:lookahead]
    if len(initial_values) < 2:
        return "increasing"
    diffs = np.diff(initial_values)
    trend_count = sum(diffs > 0)
    if trend_count > lookahead / 2:
        return "increasing"
    else:
        return "decreasing"

def filter_trend(f_values, n_values, lookahead=3):
    '''
    Function to filter values that do not follow the initial trend
    '''
    if len(f_values) < 2:
        return f_values, n_values

    initial_trend = determine_initial_trend(f_values, lookahead)
    f_filtered_values = [f_values[0]]
    n_filtered_values = [n_values[0]]

    for i in range(1, len(f_values)):
        if initial_trend == "increasing" and f_values[i] > f_filtered_values[-1]:
            f_filtered_values.append(f_values[i])
            n_filtered_values.append(n_values[i])
        elif initial_trend == "decreasing" and f_values[i] < f_filtered_values[-1]:
            f_filtered_values.append(f_values[i])
            n_filtered_values.append(n_values[i])

    return f_filtered_values, n_filtered_values


def eval_n_eff_balayage_k(X_without_k, f_desired, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Function to evaluate the effective index (n_eff) at a desired frequency using the feedforward model
    and the function given by calculate_func_neff_f
    Input: X_without_k = [w, DC, pitch], f_desired
    Output: y(f_desired) with the function y(f) = n deduced from calculate_func_neff_f
    '''
    n_k_list, f_k_list = calculate_func_neff_f(X_without_k, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    f_filtered, n_filtered = filter_trend(f_k_list, n_k_list)
    if f_filtered == []:
        return 0
    coefficients = np.polyfit(f_filtered, n_filtered, 5)
    n_eff_eval = np.polyval(coefficients, f_desired)
    return n_eff_eval