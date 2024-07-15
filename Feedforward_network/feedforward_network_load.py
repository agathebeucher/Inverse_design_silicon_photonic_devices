import torch
from .feedforward_network_model import FeedForwardNN


def feedforward_network_load(device):
    # FEEDFORWARD MODEL
    input_size_ffn = 4
    output_size_ffn = 5000
    best_hidden_sizes_ffn = [955, 925, 1005, 407, 580, 1309]
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn).to(device)
    feedforward_model.load_state_dict(torch.load('feedforward_model_trained_gpu_5000.pth'), map_location=torch.device('cpu'))
    feedforward_model.to(torch.device('cpu'))
    feedforward_model.eval()