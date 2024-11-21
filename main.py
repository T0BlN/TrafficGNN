from data_preparation import create_graph
from visualize_graph import visualize_graph
from model import TrafficGNN
from train import train_model, test_model
from visualize import visualize_results
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

data, nx_graph = create_graph()

visualize_graph(nx_graph)

num_edges = data.edge_attr.size(0)
perm = torch.randperm(num_edges)
train_mask = perm[:int(0.6 * num_edges)]
test_mask = perm[int(0.6 * num_edges):]

model = TrafficGNN(
    node_in_channels=data.x.size(1), 
    edge_in_channels=data.edge_attr.size(1), 
    hidden_channels=32,
    out_channels=1
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

train_model(model, data, train_mask, optimizer, epochs=300)

pred, loss = test_model(model, data, test_mask)

visualize_results(pred, data.edge_attr, test_mask)

plt.show()



#Run python "python main.py" to generate the prediction and node graphs