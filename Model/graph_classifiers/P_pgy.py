import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from DiffPool import DiffPool

edge_index_1 = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])
x_1 = torch.randn(5, 2)  # 5 nodes.
edge_attr1=torch.randn(len(x_1),1)

edge_index_2 = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])
x_2 = torch.randn(4, 2)  # 4 nodes.
edge_attr2=torch.ones(len(x_2),1)

edge_index_3 = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x_3 = torch.randn(4, 2)
edge_attr3=torch.ones(len(x_3),1)

data1= Data(x=x_1,edge_index=edge_index_1,y=0,edge_attr=edge_attr1)
data2= Data(x=x_2,edge_index=edge_index_2,y=0,edge_attr=edge_attr2)
data3= Data(x=x_3,edge_index=edge_index_3,y=1,edge_attr=edge_attr3)
#上面是构建3张Data图对象
data_list = [data1, data2,data3]
data = Batch.from_data_list(data_list)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
data=data.to(device)

model = DiffPool(2,2).to(device)
print(model)

out=model(data)
print('out:\n',out)

