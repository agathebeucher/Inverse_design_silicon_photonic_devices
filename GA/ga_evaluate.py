import torch
import scipy.constants as ct
import numpy as np
from ./ga_model import ga
from ./balayage_k import n_eff_one
from ../EDA/normalize_data.py import X_data_array_50_mean, X_data_array_50_std, filtered_frequencies


def eval_neff_params_frequency(best_params, desired_neff, desired_frequency):
    '''
    Evalue l'indice effectif prédit par le feed_forward network avec les meilleurs paramètres w,DC,pitch,k, 
    et le k déduit du n voulu et de la fréquence choisie
    input : w,DC, pitch, f_desired, n_desired
    output : n_pred, f_pred, spectrum_pred
    '''
    k_desired=desired_neff*(2*np.pi)*desired_frequency/ct.c
    k_desired_normalized=(k_desired-X_data_array_50_mean[3])/X_data_array_50_std[3]
    best_params_plus_k=best_params+[k_desired_normalized]
    best_params_tensor = torch.tensor(np.array(best_params_plus_k), dtype=torch.float32).to(device)
    response_np=feedforward_model(best_params_tensor).to(device).detach().cpu().numpy()     
    best_params_denormalized=best_params_plus_k*X_data_array_50_std+X_data_array_50_mean
    n_response=n_eff_one(best_params_denormalized, response_np, filtered_frequencies)
    return(n_response, response_np)

def error_npred_ndesired(desired_neff, desired_frequency):
    '''
    Evalue l'erreur entre l'indice voulu et l'indice prédit par le feed_forward network à ces paramètres
    '''
    best_ind = ga(desired_frequency, desired_neff)
    n_response, response_np=eval_neff_params_frequency(best_ind, desired_neff, desired_frequency)
    if len(n_response[0])==0:
        return(0, response_np, best_ind)
    error=abs(n_response[0][0]-desired_neff)
    return(error, response_np, best_ind, n_response[0], n_response[1])

def error_npred_ndesired(best_ind, desired_neff, desired_frequency):
    '''
    Evalue l'erreur entre l'indice voulu et l'indice prédit par le feed_forward network à ces paramètres
    '''
    best_ind = ga(desired_frequency, desired_neff)
    n_response, response_np=eval_neff_params_frequency(best_ind, desired_neff, desired_frequency)
    if len(n_response[0])==0:
        return(0, response_np, best_ind)
    error=abs(n_response[0][0]-desired_neff)
    return(error, response_np, best_ind, n_response[0], n_response[1])

