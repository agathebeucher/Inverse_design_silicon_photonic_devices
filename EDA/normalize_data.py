import numpy as np
from scipy.signal import find_peaks

X_data_array_50_mean=[4.1900793e+02, 5.5103260e-01, 4.4058459e+02, 5.7011140e+06]
X_data_array_50_std=[1.1512772e+02, 2.1334822e-01, 1.2148594e+02, 1.5868348e+06]
filtered_frequencies=np.linspace(171309976000000, 222068487407407, 50)
frequencies = np.linspace(171309976000000, 222068487407407, 5000)

def denormalize_X(X_data_array):
    N= len(X_data_array)
    return(X_data_array*X_data_array_50_std[:N]+X_data_array_50_mean[:N])

def normalize_X(X_data_array):
    print("Normalizing X data")
    # X_data est l'ensemble de données d'entrée
    X_data_array_mean = np.mean(X_data_array, axis=0)  # Calculer la moyenne de chaque paramètre
    X_data_array_std = np.std(X_data_array, axis=0)    # Calculer l'écart type de chaque paramètre

    # Normalisation des données d'entrée
    X_data_array_normalized = (X_data_array - X_data_array_mean) / X_data_array_std

    return(X_data_array_normalized)

def normalize_y(y_data_array):
    print("Normalizing y data")
    def normalize_with_max_peak(data_array, window_size=None, adjust_values=True):
        num_samples, vector_length = data_array.shape
        normalized_data_array = np.zeros_like(data_array)
        max_amplitudes = np.zeros(num_samples)

        for i in range(num_samples):
            # Trouver les pics dans le vecteur actuel
            peaks, _ = find_peaks(data_array[i])
            
            # Vérifier si le premier ou le dernier point est un pic local
            if data_array[i, 0] > data_array[i, 1]:
                peaks = np.append(peaks, 0)
            if data_array[i, -1] > data_array[i, -2]:
                peaks = np.append(peaks, vector_length - 1)

            # Trouver l'amplitude maximale parmi les pics
            if len(peaks) > 0:
                max_amplitude = np.max(data_array[i, peaks])
            else:
                max_amplitude = np.max(data_array[i])

            # Stocker l'amplitude maximale
            max_amplitudes[i] = max_amplitude

            # Normaliser le signal par l'amplitude maximale
            if max_amplitude > 0:
                normalized_data_array[i] = data_array[i] / max_amplitude
            else:
                normalized_data_array[i] = data_array[i]

            # Réajuster les valeurs normalisées si nécessaire pour conserver les variations relatives
            if adjust_values and window_size:
                for peak in peaks:
                    start_idx = max(0, peak - window_size // 2)
                    end_idx = min(vector_length, peak + window_size // 2)
                    segment = normalized_data_array[i, start_idx:end_idx]
                    normalized_data_array[i, start_idx:end_idx] = segment * max_amplitude

        return normalized_data_array, max_amplitudes

    y_data_array_normalized, y_data_array_max_amplitudes = normalize_with_max_peak(y_data_array)

    return y_data_array_normalized, y_data_array_max_amplitudes
