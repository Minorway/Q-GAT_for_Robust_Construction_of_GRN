import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
import sys,os
sys.path.append('%s/graph_classifiers/' % os.path.dirname(os.path.realpath(__file__)))
from torch.nn import Linear,Sequential, ReLU

class NeuralNetwork(nn.Module):
    def __init__(self,with_dropout=False):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.with_dropout = with_dropout
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
        )

    def forward(self, x):
        x = self.flatten(x)
        if self.with_dropout:
            x = F.dropout(x, p=0.5,training=self.training)
        logits = self.linear_relu_stack(x)
        return logits

class MLPClassifier(torch.nn.Module):
    def __init__(self, dim_features=32,hidden_units = 128, dim_target=2,with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.fc_global = Linear(dim_features, hidden_units)
        self.out = Linear(hidden_units, dim_target)
        self.with_dropout = with_dropout


    def forward(self, x):
        x1=x
        x=F.relu(self.fc_global(x1))
        if self.with_dropout:
            x = F.dropout(x,p=0.5,training=self.training)
        logits=self.out(x)
        logits=F.softmax(logits, dim=-1)

        return logits

class MLP(torch.nn.Module):

    def __init__(self, dim_features=32, hidden_units = 128,dim_target=2):
        super(MLP, self).__init__()

        self.fc_global = Linear(dim_features, hidden_units)
        self.out = Linear(hidden_units, dim_target)

       # self.fc1 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
       # self.fc2 = nn.Linear(in_features=hidden_units, out_features=dim_target)

       # self.mlp = Sequential(Linear(32, 128), ReLU(), Linear(128, 128), ReLU(), Linear(128, 2))
       # self.mlp2 = Sequential(Linear(32, 128),ReLU(), Linear(128, 2))

    def forward(self, x):

        '''
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        '''

       # out = self.fc2(torch.relu(self.fc1(x)))

        return self.out(F.relu(self.fc_global(x)))

if __name__ == '__main__':
    model=MLPClassifier()
    print(model)