import torch
import torch.nn as nn

# Définir votre modèle inverse
class InverseNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(InverseNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Première couche
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        # Couches cachées
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        # Dernière couche
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x