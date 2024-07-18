import ast 
import pandas as pd
import csv
import numpy as np

def load_data():
    
    print("Filtering data...")
    
    # LOAD pre-processed data with only one peak and 5000 values
    X_data_array_5000=np.load('/home/beucher/Documents/PRE/PRE/data/X_data_array_5000.npy')
    y_data_array_5000=np.load('/home/beucher/Documents/PRE/PRE/data/y_data_array_5000.npy')

    return(X_data_array_5000, y_data_array_5000)