import torch
import torch.nn.functional as F

def train_model(model, data, train_mask, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        out = out.squeeze()
        loss = F.mse_loss(out[train_mask], data.edge_attr[train_mask].squeeze())
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

def test_model(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.squeeze()
        loss = F.mse_loss(pred[test_mask], data.edge_attr[test_mask].squeeze())

        print("Test Loss:", loss.item())
        print("Sample predictions:", pred[test_mask][:5].cpu().numpy())
        print("Sample actual values:", data.edge_attr[test_mask][:5].cpu().numpy())

        return pred, loss