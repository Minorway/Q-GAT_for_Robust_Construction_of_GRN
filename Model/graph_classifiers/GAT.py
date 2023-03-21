import argparse
import time
from tqdm import tqdm
import torch
import copy as cp
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv


"""

GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder; 
the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional 
comment word2vec (spaCy) embeddings plus 10-dimensional profile features 

Paper: Fake News Detection on Social Media using Geometric Deep Learning
Link: https://arxiv.org/pdf/1902.06673.pdf


Model Configurations:

Vanilla GCNFN: args.concat = False, args.feature = content
UPFD-GCNFN: args.concat = True, args.feature = spacy

"""


class GAT(torch.nn.Module):
	def __init__(self, num_features,nhid,num_classes,concat=False):
		super(GAT, self).__init__()

		self.num_features = num_features
		self.num_classes = num_classes
		self.nhid = nhid
		self.concat = concat

		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

		self.fc1 = Linear(self.nhid * 2, self.nhid)

		if self.concat:
			self.fc0 = Linear(self.num_features, self.nhid)
			self.fc1 = Linear(self.nhid * 2, self.nhid)

		self.fc2 = Linear(self.nhid, self.num_classes)


	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch
		#print('batch',batch)

		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		print("beforepooling",x.shape)
		x = F.selu(global_mean_pool(x, batch))
		print("afterpooling",x.shape)
		x = F.selu(self.fc1(x))
		print("beforedropout",x.shape)
		x = F.dropout(x, p=0.5, training=self.training)
		print("afterdropout",x.shape)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.fc0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.fc1(x))

		#x = F.log_softmax(self.fc2(x), dim=-1)
		features = x
		x = self.fc2(x)

		return x ,features

