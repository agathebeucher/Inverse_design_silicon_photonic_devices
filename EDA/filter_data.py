import ast 
import pandas as pd
import csv
import numpy as np

def filter_data():
    
    #print("Importing data...")
    #file_path = "/home/beucher/Documents/PRE/PRE/data/NN_training_combine_new.csv"
    #df= pd.read_csv(file_path)

    #print("Converting data...")
    # Convertir chaque élément de la colonne 'E' en une liste de float
    #df['E_5000'] = df['E_5000'].apply(lambda x: ast.literal_eval(x))
    #y_data_50 = df['E_50']
    #y_data_50.fillna('', inplace=True)
    #y_data_50 = y_data_50.apply(lambda x: ast.literal_eval(x)if x != '' else None)
    
    print("Filtering data...")
    # Diviser les données en X (paramètres) et y (nombre de pics)
    #X_data_5000= df[['w', 'DC', 'pitch', 'k']]
    #y_data_5000= df['E_5000']
    #X_data_500= df[['w', 'DC', 'pitch', 'k']]
    #y_data_500= df['E_500']
    #X_data_50 = df[['w', 'DC', 'pitch', 'k']]
    #y_data_50 = df['E_50']
    #nb_peaks=df['nombre_de_pics_50']

    # Filtrer les données pour conserver seulement celles ayant au moins un pic
    #indices_at_least_one_peak_5000 = df[df['nombre_de_pics_5000'] >= 1].index
    #X_data_at_least_one_peak_5000 = X_data_5000.loc[indices_at_least_one_peak_5000] # Données à 5000 points
    #y_data_at_least_one_peak_5000 = y_data_5000.loc[indices_at_least_one_peak_5000]

    #indices_at_least_one_peak_500 = df[df['nombre_de_pics_500'] >= 1].index
    #X_data_at_least_one_peak_500 = X_data_500.loc[indices_at_least_one_peak_500] # Données réduites à 500 points
    #y_data_at_least_one_peak_500 = y_data_500.loc[indices_at_least_one_peak_500]


    X_data_5000=np.load('/home/beucher/Documents/PRE/PRE/data/X_data_array_5000.npy').tolist()
    y_data_5000=np.load('/home/beucher/Documents/PRE/PRE/data/y_data_array_5000.npy').tolist()
    nb_peaks_5000=np.load('/home/beucher/Documents/PRE/PRE/data/nb_peaks_array_5000.npy').tolist()

    indices_at_least_one_peak_50 = [i for i, val in enumerate(nb_peaks_5000) if val == 1.0]
    X_data_one_peak_5000 = [X_data_5000[i] for i in indices_at_least_one_peak_50]
    y_data_one_peak_5000 = [y_data_5000[i] for i in indices_at_least_one_peak_50]
    #nb_peaks_at_least_one_peak_50 = [nb_peaks[i] for i in indices_at_least_one_peak_50]
    #y_data_at_least_one_peak_5000 = y_data_5000.loc[indices_at_least_one_peak_50]

    #X_data_array_5000= np.array(X_data_at_least_one_peak_5000.values, dtype=np.float32)
    #y_data_array_5000 = np.array(y_data_at_least_one_peak_5000.values.tolist(), dtype=np.float32)
    #X_data_array_500= np.array(X_data_at_least_one_peak_500.values, dtype=np.float32)
    #y_data_array_500 = np.array(y_data_at_least_one_peak_500.values.tolist(), dtype=np.float32)
    X_data_array_5000= np.array(X_data_one_peak_5000, dtype=np.float32)
    y_data_array_5000= np.array(y_data_one_peak_5000, dtype=np.float32)

    return(X_data_array_5000, y_data_array_5000)