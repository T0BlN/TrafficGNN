import torch
import torch.nn.functional as F

def train_model(model, data, train_mask, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        out = out.squeeze()
        loss = F.smooth_l1_loss(out[train_mask], data.edge_attr[train_mask].squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        if (epoch + 1) % 10 == 0:
            print(f"Sample Predictions (Epoch {epoch + 1}):", out[train_mask][:5].detach().cpu().numpy())

def test_model(model, data, test_mask):
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr)
        pred = pred.squeeze()
        loss = F.mse_loss(pred[test_mask], data.edge_attr[test_mask].squeeze())

        return pred, loss