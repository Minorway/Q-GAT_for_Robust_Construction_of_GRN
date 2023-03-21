import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.utils import softmax, to_dense_batch


class PiNet(Module):

    def __init__(self, num_feats=2, dims=None, num_classes=2, message_passing='GCN', GCN_improved=False,
                 GAT_heads=None, skip=True):
        super(PiNet, self).__init__()
        if dims is None:
            dims = [64, 64]
        self.skip = skip
        if message_passing == 'GCN':
            from torch_geometric.nn import GCNConv

            self.mp_a1 = GCNConv(num_feats, dims[0], improved=GCN_improved)
            self.mp_x1 = GCNConv(num_feats, dims[0], improved=GCN_improved)

            skip_dims = dims[0] + num_feats if skip else dims[0]
            self.linear_a = torch.nn.Linear(skip_dims, dims[0])
            self.linear_x = torch.nn.Linear(skip_dims, dims[0])

            self.mp_a2 = GCNConv(dims[0], dims[1], improved=GCN_improved)
            self.mp_x2 = GCNConv(dims[0], dims[1], improved=GCN_improved)

            skip_dims2 = dims[1] + dims[0] if skip else dims[1]
            self.linear2 = torch.nn.Linear(skip_dims2 ** 2, num_classes)

        elif message_passing == 'GAT':
            from torch_geometric.nn import GATConv

            if (GAT_heads == None):
                GAT_heads = [5, 2]
            self.mp_a1 = GATConv(num_feats, dims[0], heads=GAT_heads[0])
            self.mp_x1 = GATConv(num_feats, dims[0], heads=GAT_heads[0])

            skip_dims = (dims[0] * GAT_heads[0]) + num_feats if skip else dims[0] * GAT_heads[0]

            self.linear_a = torch.nn.Linear(skip_dims, dims[0])
            self.linear_x = torch.nn.Linear(skip_dims, dims[0])

            self.mp_a2 = GATConv(dims[0] if skip else dims[0] * GAT_heads[0], dims[1], heads=GAT_heads[1])
            self.mp_x2 = GATConv(dims[0] if skip else dims[0] * GAT_heads[0], dims[1], heads=GAT_heads[1])

            skip_dims2 = (dims[1] * GAT_heads[1]) + dims[0] if skip else dims[1] * GAT_heads[1]

            self.linear2 = torch.nn.Linear(skip_dims2 ** 2, num_classes)

        elif message_passing == 'SAGE':
            from torch_geometric.nn import SAGEConv

            self.mp_a1 = SAGEConv(num_feats, dims[0])
            self.mp_x1 = SAGEConv(num_feats, dims[0])

            skip_dims = dims[0] + num_feats if skip else dims[0]
            self.linear_a = torch.nn.Linear(skip_dims, dims[0])
            self.linear_x = torch.nn.Linear(skip_dims, dims[0])

            self.mp_a2 = SAGEConv(dims[0], dims[1])
            self.mp_x2 = SAGEConv(dims[0], dims[1])

            skip_dims2 = dims[1] + dims[0] if skip else dims[1]
            self.linear2 = torch.nn.Linear(skip_dims2 ** 2, num_classes)


    def forward(self, data): #data.num_graphs
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs

        a_1 = F.relu(self.mp_a1(x, edge_index))
        x_1 = F.relu(self.mp_x1(x, edge_index))
        if self.skip:
            a_1 = torch.cat([a_1, x], dim=1)
            x_1 = torch.cat([x_1, x], dim=1)
            a_1 = self.linear_a(a_1)
            x_1 = self.linear_x(x_1)


        a_2 = self.mp_a2(a_1, edge_index)
        x_2 = F.relu(self.mp_x2(x_1, edge_index))

        if self.skip:
            a_2 = torch.cat([a_2, a_1], dim=1)
            x_2 = torch.cat([x_2, x_1], dim=1)

        a_2 = softmax(a_2, batch)

        a_batch, _ = to_dense_batch(a_2, batch)
        a_t = a_batch.transpose(2, 1)
        x_batch, _ = to_dense_batch(x_2, batch)
        prods = torch.bmm(a_t, x_batch)
        flat = torch.flatten(prods, 1, -1)

        features = flat
        batch_out = self.linear2(flat)

        #final = F.softmax(batch_out, dim=-1)
        return batch_out,features
