import torch
import scipy.constants as ct
import numpy as np
import sys
import os
from .ga_model import ga, eval_neff_params_frequency
from .balayage_k import n_eff_one

def error_npred_ndesired(desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Function to evaluate the error between the desired effective index and the predicted index by the feedforward network for given parameters
    '''
    # Use a genetic algorithm to find the best parameters for the desired frequency and effective index
    best_ind = ga(desired_frequency, desired_neff, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    
    # Evaluate the predicted effective index for the best parameters
    n_response, response_np = eval_neff_params_frequency(best_ind, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    
    # If no response is found, return 0 error
    if len(n_response[0]) == 0:
        return 0, response_np, best_ind
    
    # Calculate the absolute error between the predicted and desired effective index
    error = abs(n_response[0][0] - desired_neff)
    
    # Return the error, the predicted response, the best parameters, and the predicted effective index and frequency
    return error, response_np, best_ind, n_response[0], n_response[1]


def error_npred_ndesired(best_ind, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Overloaded function to evaluate the error between the desired effective index and the predicted index by the feedforward network for given parameters
    '''
    # Evaluate the predicted effective index for the best parameters
    n_response, response_np = eval_neff_params_frequency(best_ind, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    # If no response is found, return 0 error
    if len(n_response[0]) == 0:
        return 0, response_np, best_ind
    # Calculate the absolute error between the predicted and desired effective index
    error = abs(n_response[0][0] - desired_neff)
    # Return the error, the predicted response, the best parameters, and the predicted effective index and frequency
    return error, response_np, best_ind, n_response[0], n_response[1]

