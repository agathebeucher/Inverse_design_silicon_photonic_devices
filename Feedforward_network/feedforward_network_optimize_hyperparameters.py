import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .feedforward_network_model import FeedForwardNN
from EDA.load_data import load_data, create_datasets
from EDA.normalize_data import normalize_X, normalize_y
from sklearn.model_selection import KFold, train_test_split

# You can find an exemple of how to load/normalize/divide the data into datasets in the main.py file

def optmize_hyperparameters_FFN(X_train_5000_normalized, y_train_5000_normalized):
    '''
    The following function is used to optimize the hyperparameters of a feedforward neural network using Optuna
    '''
    # Define the objective function for Optuna with cross-validation
    def objective(trial):
        n_fc_layers = trial.suggest_int("n_fc_layers", 1, 5)  # Number of fully-connected layers
        hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 50, 500) for i in range(n_fc_layers)]
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        
        # Cross validation to evaluate hyperparameter performance
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_losses = []
        
        for train_index, val_index in kf.split(X_train_5000_normalized):
            X_tr, X_val = X_train_5000_normalized[train_index], X_train_5000_normalized[val_index]
            y_tr, y_val = y_train_5000_normalized[train_index], y_train_5000_normalized[val_index]

            X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
            y_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
            X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

            input_size = X_tr.shape[1] 
            output_size = y_tr.shape[1]

            model = FeedForwardNN(input_size, hidden_sizes, output_size).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            for epoch in range(500):  # Choose the number of epochs for optimization
                model.train()
                optimizer.zero_grad()
                outputs = model(X_tr)
                loss = criterion(outputs, y_tr)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                fold_losses.append(val_loss.item())

        return np.mean(fold_losses)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an optuna study and start optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

    # Best hyperparameters found by Optuna
    best_trial = study.best_trial

    return best_trial.params