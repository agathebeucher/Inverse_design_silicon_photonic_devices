import torch
import torch.nn as nn

# Défine the Neural Network model
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList() # Initialize an empty list to store layers
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0])) # Input layer to the first hidden layer
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x)) # Apply ReLU activation function to each layer's output
        x = self.layers[-1](x) # For the last layer, do not apply activation function
        return x  # Return the final output