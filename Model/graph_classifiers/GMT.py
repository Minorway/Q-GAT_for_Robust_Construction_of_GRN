import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv 
from torch_geometric.nn.glob.gmt import GraphMultisetTransformer



class GMT(torch.nn.Module):
    def __init__(self,num_features=2,num_classes=2):
        super().__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(
            in_channels=96,
            hidden_channels=64,
            out_channels=32,
            Conv=GCNConv,
            num_nodes=13,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False,
        )

        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, num_classes)

    def forward(self, data):
        x0, edge_index, batch = data.x ,data.edge_index,data.batch

        x1 = self.conv1(x0, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        x3 = self.conv3(x2, edge_index).relu()
        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.pool(x, batch, edge_index)

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        features = x
        x = self.lin2(x)

        return x ,features

