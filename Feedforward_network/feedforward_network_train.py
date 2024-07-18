import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from .feedforward_network_model import FeedForwardNN


def feedforward_network_train(X_train, y_train, X_val, y_val):
    # Network parameters
    input_size = X_train.shape[1] # Number of input features
    output_size = y_train.shape[1] # Number of output targets

    # Best optimized parameters with Optuna for the network
    best_lr = 0.00067763307084440112
    best_hidden_sizes = [955, 925, 1005, 407, 580, 1309]

    # Initialize the feedforward model with the best parameters
    feedforward_model = FeedForwardNN(input_size, best_hidden_sizes, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(feedforward_model.parameters(), lr=best_lr)

    # Lists to store the training and validation losses to plot them later
    train_losses = []
    val_losses = []

    num_epochs = 3400 # Total number of training epochs
    for epoch in range(num_epochs): 
        feedforward_model.train() # Set the model to training mode
        optimizer.zero_grad() # Clear the gradients
        outputs = feedforward_model(X_train)  # Forward pass on the training data
        loss = criterion(outputs, y_train)  # Compute the training loss
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update the model parameters

        # Stocker la perte d'entraînement
        train_losses.append(loss.item())

        if epoch % 100 == 0:
            # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation
                val_outputs = feedforward_model(X_val)
                val_loss = criterion(val_outputs, y_val)
                # Stocker la perte de validation
                val_losses.append(val_loss.item())
                print(f"Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    # Plot the training and validation losses over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(range(0, num_epochs, 100), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs for Response prediction network')
    plt.legend()
    plt.savefig("../results/Train_Val_loss_over_Epochs_Response_network.png")
    #plt.show()

    # Sauvegarder le modèle
    #torch.save(feedforward_model.state_dict(), 'feedforward_model_trained.pth')

    return(feedforward_model) # return the trained model
