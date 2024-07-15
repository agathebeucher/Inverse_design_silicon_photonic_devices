import numpy as np
from scipy.signal import argrelextrema
from scipy import constants as ct
from .approx_gauss import approx_gauss
import torch

def conversion_k(k_usi, grating_pitch):
    return(2*np.pi*k_usi/grating_pitch)

def find_1D_localmax_one(data,threshold, sample_dist):
    idx = argrelextrema(data=data, comparator=np.greater, order=sample_dist)[0]
    idx=idx[np.where(data[idx]>threshold)]
    max_value = max(data)
    maxima_amplitude = data[idx]
    maxima_amplitude_sorted = np.sort(maxima_amplitude)
    maxima_amplitude_sorted = np.flipud(maxima_amplitude_sorted)
    c = np.empty(maxima_amplitude_sorted.size, dtype=np.int64)
    for i in range(maxima_amplitude_sorted.size):
        temp = maxima_amplitude.tolist().index(maxima_amplitude_sorted[i]);
        c[i] = idx[temp]
    if len(c)>0:
        if max_value-data[c[0]]>0.5:
            return([])
        else:
            return(c[0])
    else:
        return([])

def n_eff_one(X_data, y_data, freq):
    #get analysis parameters
    peaks_threshold = 0.1
    sample_dist = 1

    #load data
    freq_axis= freq
    grating_pitch = X_data[2]*1e-9
    k_value = X_data[3]
    fs_all = y_data

    #find peaks in the spectrum
    peaks_idx = find_1D_localmax_one(fs_all,peaks_threshold,sample_dist)
    if len(peaks_idx)==1:
        f_band=freq_axis[peaks_idx]
        bloch_index=k_value*ct.c/2/np.pi/f_band;
        return([bloch_index[0]], [f_band[0]])
    else :
        return([],[])

def calculate_func_neff_f(X_params_without_k_normalized, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    input : X_params_without_k_normalized = [w, DC, pitch]
    output : (n_k_list, f_k_list) -> les points de la courbe n=y(f)
    '''
    grating_pitch_denormalized=X_params_without_k_normalized[2]*X_data_array_5000_std[2]+X_data_array_5000_mean[2]
    k_list_denormalized = [conversion_k(round(num, 2), grating_pitch_denormalized*1e-9) for num in np.arange(0.25, 0.50, 0.01)]
    k_list_normalized = [(k_denormalized-X_data_array_5000_mean[3])/X_data_array_5000_std[3] for k_denormalized in k_list_denormalized]
    X_params_list = np.array([X_params_without_k_normalized+[k] for k in k_list_normalized])
    spectrum_k_list=[]
    n_k_list=[]
    f_k_list=[]
    k_list_peak=[]
    n_k_list_temp=[]
    f_k_list_temp=[]
    # Prédire les spectre en fréquence pour chaque k
    for i in range(len(k_list_normalized)):
        X_params_tensor = torch.tensor(np.array(X_params_list[i]), dtype=torch.float32).to(device)
        spectrum_k=feedforward_model(X_params_tensor).to(device)
        spectrum_k_list.append(spectrum_k.detach().cpu().numpy())
        '''
        mse, y_fit=approx_gauss(spectrum_k.detach().cpu().numpy())   
        plt.rcParams["figure.figsize"]= 5,5
        plt.title(f'{n_eff_one(X_params_list[i]*X_data_array_5000_std+X_data_array_5000_mean, spectrum_k.detach().cpu().numpy(), frequencies)[1]}, mse {mse}, k={X_params_list[i][3]*X_data_array_5000_std[3]+X_data_array_5000_mean[3]}')
        plt.plot(spectrum_k.detach().cpu().numpy())
        if y_fit is not None:
            plt.plot(y_fit)
        plt.show()
        '''
    # Prédire les fréquences de résonances et les indices eff pour chaque k
    for i in range(len(spectrum_k_list)) :
        X_params_denormalized=X_params_list[i]*X_data_array_5000_std+X_data_array_5000_mean 
        n_eff_k, f_k=n_eff_one(X_params_denormalized, spectrum_k_list[i], frequencies)
        n_k_list_temp.append(n_eff_k)
        f_k_list_temp.append(f_k)
    for i in range(len(n_k_list_temp)):
        if n_k_list_temp[i] != []:
            mse, y_fit=approx_gauss(spectrum_k_list[i])   
            if i!=0 and i!= len(n_k_list_temp)-1:
                if (n_k_list_temp[i+1]!=[] or n_k_list_temp[i-1]!=[]) and y_fit is not None:
                    if mse<0.0001:
                        n_k_list.append(n_k_list_temp[i][0])
                        f_k_list.append(f_k_list_temp[i][0])
            else : 
                if y_fit is not None :
                    if mse<0.0001 :
                        n_k_list.append(n_k_list_temp[i][0])
                        f_k_list.append(f_k_list_temp[i][0])
    return(n_k_list, f_k_list)

def determine_initial_trend(values, lookahead=5):
    """
    Détermine la tendance initiale en regardant les premières valeurs.
    """
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
    """
    Filtre les valeurs qui ne suivent pas la tendance initiale.
    """
    if len(f_values) < 2:
        return f_values, n_values

    # Déterminer la tendance initiale
    initial_trend = determine_initial_trend(f_values, lookahead)
    f_filtered_values = [f_values[0]]
    n_filtered_values = [n_values[0]]
    
    # Filtrer les valeurs
    for i in range(1, len(f_values)):
        if initial_trend == "increasing" and f_values[i] > f_filtered_values[-1] :
            f_filtered_values.append(f_values[i])
            n_filtered_values.append(n_values[i])
        elif initial_trend == "decreasing" and f_values[i] < f_filtered_values[-1] :
            f_filtered_values.append(f_values[i])
            n_filtered_values.append(n_values[i])
    
    return f_filtered_values, n_filtered_values

def eval_n_eff_balayage_k(X_without_k, f_desired, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies):
    '''
    input : X_without_k = [w,DC, pitch], f_desired
    output : y(f_desired), avec le fonction y(f)=n déduit de calculate_func_neff_f
    '''
    n_k_list, f_k_list=calculate_func_neff_f(X_without_k, feedforward_model, device, X_data_array_5000_std, X_data_array_5000_mean, frequencies)
    f_filtered, n_filtered = filter_trend(f_k_list,n_k_list)
    if f_filtered==[]:
        return(0)
    coefficients = np.polyfit(f_filtered, n_filtered, 5)
    n_eff_eval = np.polyval(coefficients, f_desired)
    return(n_eff_eval)
