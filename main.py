from data_preparation import create_graph
from model import TrafficGNN
from train import train_model, test_model
from visualize import visualize_results
import torch

data = create_graph()
num_edges = data.edge_attr.size(0)

perm = torch.randperm(num_edges)
train_mask = perm[:int(0.6 * num_edges)]
test_mask = perm[int(0.6 * num_edges):]

print("Train mask indices:", train_mask)
print("Test mask indices:", test_mask)
print("Max index in train mask:", train_mask.max())
print("Max index in test mask:", test_mask.max())

model = TrafficGNN(
    node_in_channels=data.x.size(1), 
    edge_in_channels=data.edge_attr.size(1), 
    hidden_channels=16, 
    out_channels=1
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_model(model, data, train_mask, optimizer, epochs=100)
pred, loss = test_model(model, data, test_mask)

visualize_results(pred, data.edge_attr, test_mask)



#Run python "python main.py" to generate the prediction graph