import torch
from torch_geometric.nn import GCNConv #GCN layer used to aggregate and update node features by combining them with neighboring nodes
import torch.nn.functional as F

#inherits from torch.nn.Module
class TrafficGNN(torch.nn.Module):
    #initialezes with all the channels
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels):
        super(TrafficGNN, self).__init__()
        #processes the input features into a large vector respresentation using GCNConv
        self.conv1 = GCNConv(node_in_channels, hidden_channels)
        #refines with the second and third layers
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # a regularization technique used to prevent overfitting by randomly drop some features during training
        self.dropout = torch.nn.Dropout(p=0.3)
        
        #Combines the representations of the source node, destination node, and edge features to compute the final predicted ETA
        self.edge_fc = torch.nn.Sequential(
            #input is source and destination nodes (hidden_channels*2) and original edge features (ETA). Output size is hidden_channels
            torch.nn.Linear(hidden_channels * 2 + edge_in_channels, hidden_channels),
            #Non-linear activation to introduce complexity to the model
            torch.nn.ReLU(),
            #maps to desires output size (1)
            torch.nn.Linear(hidden_channels, out_channels)
        )
        """
        the function introduces non-linearity by zeroing out negative values. This non-linearity is essential for neural networks because:
            Without it, multiple layers would collapse into a single linear transformation (i.e., no matter how many layers,
            the entire network would behave like one linear layer).
            Non-linear activation functions enable the network to learn complex patterns and decision boundaries.
        """

    #defines how input data flows through the model during training
    def forward(self, x, edge_index, edge_attr):
        #passes the node features (x) though each GCN layer and applies the ReLU activation to introduce non-linearity
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        #droput to regularize
        x = self.dropout(x)
        #the final x contains updated node embeddings with hidden_channels dimensions

        #combines the embeddings with he edge features to compute predicitons
        src, dest = edge_index
        edge_features = torch.cat([x[src], x[dest], edge_attr], dim=1)
        #Results in a tensor (edge_features) representing each edge as a combination of its two connected nodes and its own features

        #Pass the edge_features through the fully connected network to predict the ETA
        return self.edge_fc(edge_features)