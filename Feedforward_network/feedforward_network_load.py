import torch
from .feedforward_network_model import FeedForwardNN


def feedforward_network_load(device):
    '''
    This function loads a pre-trained feedforward neural network model onto the specified device.
    '''
    # Define network parameters
    input_size_ffn = 4  # Number of input features
    output_size_ffn = 5000  # Number of output targets
    best_hidden_sizes_ffn = [955, 925, 1005, 407, 580, 1309]  # Best hidden layer sizes identified

    # Initialize the feedforward model with the defined parameters
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn).to(device)
    
    # Load the pre-trained model's state dictionary from the specified file path
    feedforward_model.load_state_dict(torch.load('/home/beucher/Documents/PRE/PRE/Feedforward_network/feedforward_model_trained_gpu_5000.pth', map_location=torch.device('cpu')))
    
    # Move the model to the CPU
    feedforward_model.to(torch.device('cpu'))

    # Set the model to evaluation mode
    feedforward_model.eval()

    return(feedforward_model) # return the loaded model