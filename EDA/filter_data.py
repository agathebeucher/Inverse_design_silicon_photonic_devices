import ast 
import pandas as pd
import numpy as np

def filter_data():
    file_path = "../data/NN_training_combine_new.csv"
    df = pd.read_csv(file_path) 
    
    # Convertir chaque élément de la colonne 'E' en une liste de float
    df['E_5000'] = df['E_5000'].apply(lambda x: ast.literal_eval(x))
    df['E_500'] = df['E_500'].apply(lambda x: ast.literal_eval(x))
    df['E_50'] = df['E_50'].apply(lambda x: ast.literal_eval(x))

    # Diviser les données en X (paramètres) et y (nombre de pics)
    X_data_5000= df[['w', 'DC', 'pitch', 'k']]
    y_data_5000= df['E_5000']
    X_data_500= df[['w', 'DC', 'pitch', 'k']]
    y_data_500= df['E_500']
    X_data_50 = df[['w', 'DC', 'pitch', 'k']]
    y_data_50 = df['E_50']
    nb_peaks=df['nombre_de_pics_50']

    # Filtrer les données pour conserver seulement celles ayant au moins un pic
    indices_at_least_one_peak_5000 = df[df['nombre_de_pics_5000'] >= 1].index
    X_data_at_least_one_peak_5000 = X_data_5000.loc[indices_at_least_one_peak_5000] # Données à 5000 points
    y_data_at_least_one_peak_5000 = y_data_5000.loc[indices_at_least_one_peak_5000]

    indices_at_least_one_peak_500 = df[df['nombre_de_pics_500'] >= 1].index
    X_data_at_least_one_peak_500 = X_data_500.loc[indices_at_least_one_peak_500] # Données réduites à 500 points
    y_data_at_least_one_peak_500 = y_data_500.loc[indices_at_least_one_peak_500]

    indices_at_least_one_peak_50 = df[df['nombre_de_pics_50'] >= 1].index
    X_data_at_least_one_peak_50 = X_data_50.loc[indices_at_least_one_peak_50] # Données réduites à 50 points
    y_data_at_least_one_peak_50 = y_data_50.loc[indices_at_least_one_peak_50]
    nb_peaks=nb_peaks.loc[indices_at_least_one_peak_50]
    y_data_at_least_one_peak_5000 = y_data_5000.loc[indices_at_least_one_peak_50]

    X_data_array_5000= np.array(X_data_at_least_one_peak_5000.values, dtype=np.float32)
    y_data_array_5000 = np.array(y_data_at_least_one_peak_5000.values.tolist(), dtype=np.float32)
    X_data_array_500= np.array(X_data_at_least_one_peak_500.values, dtype=np.float32)
    y_data_array_500 = np.array(y_data_at_least_one_peak_500.values.tolist(), dtype=np.float32)
    X_data_array_50= np.array(X_data_at_least_one_peak_50.values, dtype=np.float32)
    y_data_array_50 = np.array(y_data_at_least_one_peak_50.values.tolist(), dtype=np.float32)

    return(X_data_array_5000, y_data_array_5000, X_data_array_50, y_data_array_50)