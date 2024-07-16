import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from .feedforward_network_model import FeedForwardNN


def feedforward_network_train(X_train, y_train, X_val, y_val):
    # Paramètres du réseau
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    # Paramètres du réseau (meilleurs paramètres identifiés)
    # 0.0015904915677152046
    best_lr = 0.00067763307084440112
    best_hidden_sizes = [955, 925, 1005, 407, 580, 1309]

    # Entraîner le modèle avec les meilleurs paramètres
    feedforward_model = FeedForwardNN(input_size, best_hidden_sizes, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(feedforward_model.parameters(), lr=best_lr)

    # Listes pour stocker les pertes
    train_losses = []
    val_losses = []

    num_epochs = 3400
    for epoch in range(num_epochs):
        feedforward_model.train()
        optimizer.zero_grad()
        outputs = feedforward_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Stocker la perte d'entraînement
        train_losses.append(loss.item())

        if epoch % 100 == 0:
            feedforward_model.eval()
            with torch.no_grad():
                val_outputs = feedforward_model(X_val)
                val_loss = criterion(val_outputs, y_val)
                # Stocker la perte de validation
                val_losses.append(val_loss.item())
                print(f"Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    # Tracer les pertes
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

    return(feedforward_model)
