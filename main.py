from data_preparation import create_graph
from model import TrafficGNN
from train import train_model, test_model
from visualize import visualize_results
import torch

# Prepare data
data = create_graph()
num_edges = data.edge_index.size(1)

train_mask = torch.randperm(num_edges)[:int(0.8 * num_edges)]
test_mask = torch.randperm(num_edges)[int(0.8 * num_edges):]
data.train_mask = train_mask
data.test_mask = test_mask

# Define model and optimizer
model = TrafficGNN(data.x.size(1), 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train and test the model
train_model(model, data, train_mask, optimizer, epochs=100)
pred = test_model(model, data, test_mask)

# Visualize results
visualize_results(pred[test_mask].cpu().numpy(), data.edge_attr[test_mask].cpu().numpy())