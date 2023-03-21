import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import ARMAConv,global_add_pool, global_mean_pool
from torch.nn import Linear,Sequential, ReLU


class ARMA(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()

        hidden_dim=128

        self.conv1 = ARMAConv(num_features, hidden_dim, num_stacks=3,num_layers=2,shared_weights=True, dropout=0.25)
        self.conv2 = ARMAConv(hidden_dim, num_classes, num_stacks=3,num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

        self.pooling = global_add_pool

    def forward(self,data):
        x, edge_index ,batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)

        features=  self.pooling(x,batch)
        x = self.conv2(x, edge_index)

        x = self.pooling(x,batch)

        #self.mlp = Sequential(Linear(32, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 2))

        return x,features





