import torch
import torch.nn.functional as F #Contains loss functions like smooth_l1_loss and mse_loss

#function to train the model, epochs are like training rounds
def train_model(model, data, train_mask, optimizer, epochs=100):
    #puts the model into training mode, enabling dropouts
    model.train()
    print("TRAINING: \n")
    for epoch in range(epochs):
        #clears gradients from previous iteration
        optimizer.zero_grad()
        #Feeds the data into the GNN running a forward pass and produces predicitons
        out = model(data.x, data.edge_index, data.edge_attr)
        #reduces the dimensions to be compatible with the graph
        out = out.squeeze()
        #selects the prediction and actual ETA values for the training mask and computes the loss
        loss = F.smooth_l1_loss(out[train_mask], data.edge_attr[train_mask].squeeze())
        #Computes gradients of the loss with respect to the model's parameters using backpropagation
        loss.backward()
        #clips the gradients to prevent them from becoming too large which can derail the training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #updates the paramters/weights using the gradient and the optimizer
        optimizer.step()

        #prints losses and smaple predictions every 10 rounds
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        if (epoch + 1) % 10 == 0:
            print(f"Sample Predictions (Epoch {epoch + 1}):", out[train_mask][:5].detach().cpu().numpy())
    print("\n")

#evaluates the model on the unseen data
def test_model(model, data, test_mask):
    #sets the model to evaluation mode
    model.eval()
    #temporarily disable gradients for more computing efficiency
    with torch.no_grad():
        #generates the predictions, removes extra dimensions, and calculates loss
        pred = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.squeeze()
        loss = F.mse_loss(pred[test_mask], data.edge_attr[test_mask].squeeze())

        #Print some testing values
        print("EVALUATION: \n")
        print(f"Test Loss: {loss.item()}")
        actual_values = data.edge_attr[test_mask].squeeze()
        predicted_values = pred[test_mask]
        print("Sample Predictions vs Actual Values:")
        for i in range(min(5, len(test_mask))):
            print(f"Predicted: {predicted_values[i].item():.4f}, Actual: {actual_values[i].item():.4f}")


        return pred, loss