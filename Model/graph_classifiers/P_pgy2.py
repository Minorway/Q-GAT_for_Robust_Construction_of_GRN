import torch
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import Batch
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree
#from DeepGCN import DeeperGCN
from GNNCL import *

edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])
x_s = torch.randn(5, 2)  # 5 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])
x_t = torch.randn(4, 2)  # 4 nodes.

edge_index_3 = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x_3 = torch.randn(4, 2)

data1= Data(x=x_s,edge_index=edge_index_s,y=0)
data2= Data(x=x_t,edge_index=edge_index_t,y=0)
data3= Data(x=x_3,edge_index=edge_index_3,y=1)
#上面是构建3张Data图对象
data_list = [data1, data2,data3]
data = Batch.from_data_list(data_list)

#graph_emb = global_mean_pool(loader.x, loader.batch)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data=data.to(device)
model = Net(in_channels=2, num_classes=2).to(device)
print(model)


out=model(data)

print('out:\n',out)

