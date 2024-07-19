import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .feed_forward_network_model import feedforward_network_model
from sklearn.model_selection import KFold

# Load and normalize data

# Create datasets

# Définir la fonction objectif pour Optuna avec validation croisée
def objective(trial):
    n_fc_layers = trial.suggest_int("n_fc_layers", 1, 5)  # Nombre de couches entièrement connectées
    hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 50, 500) for i in range(n_fc_layers)]
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    
    # Validation croisée pour évaluer la performance des hyperparamètres
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []
    
    for train_index, val_index in kf.split(X_train_50_normalized):
        X_tr, X_val = X_train_50_normalized[train_index], X_train_50_normalized[val_index]
        y_tr, y_val = y_train_50_normalized[train_index], y_train_50_normalized[val_index]

        X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

        input_size = X_tr.shape[1]  # Taille de l'entrée
        output_size = y_tr.shape[1]  # Taille de la sortie

        model = FeedForwardNN(input_size, hidden_sizes, output_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(500):  # Réduire le nombre d'époques pour l'optimisation
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
  
# Assurer que X_train_50_normalized, y_train_50_normalized sont vos données normalisées
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Créer une étude Optuna et lancer l'optimisation
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)  # Ajuster le nombre d'essais si nécessaire

# Meilleurs hyperparamètres trouvés par Optuna
best_trial = study.best_trial
print("Best trial parameters:", best_trial.params)
