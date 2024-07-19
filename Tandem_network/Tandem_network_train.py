import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Feedforward_network.feedforward_network_model import FeedForwardNN
from Inverse_network_model import InverseNetwork

def Inverse_network_train(X_train_inverse, y_train_inverse, X_val_inverse, y_val_inverse):
    # Hyperparameters direct model
    input_size_ffn = 4
    output_size_ffn = 5000
    best_hidden_sizes_ffn = [300, 300, 300, 300]
    # Load the direct model (FFN) pre-trained
    feedforward_model = FeedForwardNN(input_size_ffn, best_hidden_sizes_ffn, output_size_ffn)
    feedforward_model.load_state_dict(torch.load('../../Feedforward_network/feedforward_model_trained.pth'))
    feedforward_model.eval()

    # Train the inverse model
    best_hidden_sizes_inverse = [300, 300,300,300,300]
    best_lr_inverse = 0.001
    best_structure_weight=0.95

    def custom_loss(predicted_field, X_fold, y_fold, predicted_design_params, structure_weight=0.95):
        criterion_response = nn.MSELoss()
        criterion_structure = nn.MSELoss()
        loss_response = criterion_response(predicted_field, X_fold)
        loss_structure = criterion_structure(predicted_design_params, y_fold)
        loss_total = structure_weight * loss_response + (1 - structure_weight) * loss_structure
        return loss_total

    # Hyperparameters inverse model
    input_size_inverse = 50
    output_size_inverse = 4
    inverse_model = InverseNetwork(input_size_inverse, best_hidden_sizes_inverse, output_size_inverse)

    # Loss function and criterion function
    criterion_inverse = nn.MSELoss()
    optimizer_inverse = optim.Adam(inverse_model.parameters(), lr=best_lr_inverse)

    num_epochs_inverse = 5000

    # List to stock the losses
    train_losses_inverse = []
    val_losses_inverse = []

    for epoch in range(num_epochs_inverse):
        inverse_model.train()
        optimizer_inverse.zero_grad()
        predicted_design_params = inverse_model(X_train_inverse)
        predicted_field = feedforward_model(predicted_design_params)
        loss = custom_loss(predicted_field, X_train_inverse, y_train_inverse, predicted_design_params, best_structure_weight)
        loss.backward()
        optimizer_inverse.step()
        train_losses_inverse.append(loss.item())
        if epoch % 100 == 0:
            inverse_model.eval()
            with torch.no_grad():
                predicted_design_params_val = inverse_model(X_val_inverse)
                predicted_field_val = feedforward_model(predicted_design_params_val)
                val_loss = custom_loss(predicted_field_val, X_val_inverse, y_val_inverse, predicted_design_params_val, best_structure_weight).item()
                val_losses_inverse.append(val_loss)
                print(f"Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")

    # Save the model
    #torch.save(inverse_model.state_dict(), 'inverse_model_trained.pth')

    # Plot the loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses_inverse, label='Training Loss')
    plt.plot(range(0, num_epochs_inverse, 100), val_losses_inverse, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs for inverse design')
    plt.legend()
    plt.savefig("../results/Train_Val_loss_over_Epochs_Inverse_design.png")
    #plt.show()

    return(inverse_model)