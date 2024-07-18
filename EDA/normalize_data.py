import numpy as np
from scipy.signal import find_peaks

# Mean and standard deviation arrays for the dataset
X_data_array_5000_mean=[4.1745990e+02, 5.7034820e-01, 4.2528439e+02, 5.6493375e+06]
X_data_array_5000_std=[1.1564603e+02, 2.1600698e-01, 1.2746652e+02, 1.7770392e+06]
frequencies = np.linspace(171309976000000, 222068487407407, 5000)

def denormalize_X(X_data_array):
    '''
    This function takes a normalized data array and converts it back to its original scale using predefined mean and standard deviation values.
    '''
    N= len(X_data_array)
    return(X_data_array*X_data_array_5000_std[:N]+X_data_array_5000_mean[:N])


def normalize_X(X_data_array):
    '''
    Function to normalize the parameters X_data with a z_score normalization
    '''
    print("Normalizing X data...")
    X_data_array_mean = np.mean(X_data_array, axis=0) 
    X_data_array_std = np.std(X_data_array, axis=0) 
    X_data_array_normalized = (X_data_array - X_data_array_mean) / X_data_array_std

    return(X_data_array_normalized, X_data_array_5000_mean, X_data_array_5000_std)

def normalize_y(y_data_array):
    '''
    Function to normalize the parameters y_data with a max peak normalization
        - the main peak is reduced to 1
        - the rest of the spectrum is reduced proportionally between 0 and 1
        --> scaling each sample to its maximum peak value
    '''
    print("Normalizing y data...")
    window_size=None
    adjust_values=True
    num_samples, vector_length = y_data_array.shape
    normalized_data_array = np.zeros_like(y_data_array)
    for i in range(num_samples):
        peaks, _ = find_peaks(y_data_array[i])
        # Check if first/last peak is local
        if y_data_array[i, 0] > y_data_array[i, 1]:
            peaks = np.append(peaks, 0)
        if y_data_array[i, -1] > y_data_array[i, -2]:
            peaks = np.append(peaks, vector_length - 1)
        # Find the peak with the maximum amplitude
        if len(peaks) > 0:
            max_amplitude = np.max(y_data_array[i, peaks])
        else:
            max_amplitude = np.max(y_data_array[i])
        # Normalize the data with the maximum amplitude
        if max_amplitude > 0:
            normalized_data_array[i] = y_data_array[i] / max_amplitude
        else:
            normalized_data_array[i] = y_data_array[i]
        # Adjust normalized values to preserve relative variations, if necessary
        if adjust_values and window_size:
            for peak in peaks:
                start_idx = max(0, peak - window_size // 2)
                end_idx = min(vector_length, peak + window_size // 2)
                segment = normalized_data_array[i, start_idx:end_idx]
                normalized_data_array[i, start_idx:end_idx] = segment * max_amplitude

    return normalized_data_array
