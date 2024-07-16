import torch
import scipy.constants as ct
import numpy as np
import sys
import os
from .ga_model import ga, eval_neff_params_frequency
from .balayage_k import n_eff_one

def error_npred_ndesired(desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Evalue l'erreur entre l'indice voulu et l'indice prédit par le feed_forward network à ces paramètres
    '''
    best_ind = ga(desired_frequency, desired_neff, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    n_response, response_np=eval_neff_params_frequency(best_ind, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    if len(n_response[0])==0:
        return(0, response_np, best_ind)
    error=abs(n_response[0][0]-desired_neff)
    return(error, response_np, best_ind, n_response[0], n_response[1])

def error_npred_ndesired(best_ind, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    Evalue l'erreur entre l'indice voulu et l'indice prédit par le feed_forward network à ces paramètres
    '''
    n_response, response_np=eval_neff_params_frequency(best_ind, desired_neff, desired_frequency, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    if len(n_response[0])==0:
        return(0, response_np, best_ind)
    error=abs(n_response[0][0]-desired_neff)
    return(error, response_np, best_ind, n_response[0], n_response[1])

