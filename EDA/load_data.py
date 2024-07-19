import numpy as np
from sklearn.model_selection import KFold, train_test_split

def load_data():
    
    print("Filtering data...")
    
    # LOAD pre-processed data with only one peak and 5000 values
    X_data_array_5000=np.load('/home/beucher/Documents/PRE/PRE/data/X_data_array_5000.npy')
    y_data_array_5000=np.load('/home/beucher/Documents/PRE/PRE/data/y_data_array_5000.npy')

    return(X_data_array_5000, y_data_array_5000)

def create_datasets(X_data_array_5000_normalized, y_data_array_5000_normalized, test_size):
    # Divide the dataset into train/test/val datasets
    X_train_5000_normalized, test_val_X_5000_normalized, y_train_5000_normalized, test_val_Y_5000_normalized = train_test_split(X_data_array_5000_normalized, y_data_array_5000_normalized, test_size=test_size, random_state=42)
    X_test_5000_normalized, X_val_5000_normalized, y_test_5000_normalized, y_val_5000_normalized= train_test_split(test_val_X_5000_normalized, test_val_Y_5000_normalized, test_size=0.5, random_state=42)
    return(X_train_5000_normalized, y_train_5000_normalized, X_test_5000_normalized, y_test_5000_normalized, X_val_5000_normalized, y_val_5000_normalized)
