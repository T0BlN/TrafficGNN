from data_preparation import create_graph #used to generate graph and prepare it for training
from visualize_graph import visualize_graph #used to visualize the node graph
from model import TrafficGNN #imports the class for the network
from train import train_model, test_model #imports the training and testing functions
from visualize import visualize_results #visualizes the prediction plot
import torch #for using tensors
from torch.optim.lr_scheduler import StepLR #a scheduler to adjust the learning rate during training
import matplotlib.pyplot as plt #used for plotting graphs

#creates the random graph with the function from data_preperation.py
data, nx_graph = create_graph()


#creates the visual of the graph
visualize_graph(nx_graph)


#pulls every edge in the graph and randomly chooses 60% for training and 40% for testing
num_edges = data.edge_attr.size(0)
perm = torch.randperm(num_edges)
train_mask = perm[:int(0.6 * num_edges)]
test_mask = perm[int(0.6 * num_edges):]

#initializes the model with the TrafficGNN imported class
model = TrafficGNN(
    #number of input features for each node (degree, avg speed)
    node_in_channels=data.x.size(1), 
    #number of input features for each edge (ETA)
    edge_in_channels=data.edge_attr.size(1), 
    #number of hidden units int he intermediate layers of the GNN
    hidden_channels=32,
    #the output of the model (predicted ETA)
    out_channels=1
)


#optimizes the model parameters during training (updates parameters every 0.001 seconds)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#trains the model with the imported training function
train_model(model, data, train_mask, optimizer, epochs=300)
#test the trained model on the test edges
pred, loss = test_model(model, data, test_mask)
#visualizes the results
visualize_results(pred, data.edge_attr, test_mask)

plt.show()

#Run python "python main.py" to generate the prediction and node graphs