import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as cPickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import math
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from inspect import signature
sys.path.append('%s/software/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/software/node2vec/src'  % os.path.dirname(os.path.realpath(__file__)))
cur_dir = os.path.dirname(os.path.realpath(__file__))
print("cur_dir",cur_dir)
#from main import *
sys.path.append('%s/preprocessing'% os.path.dirname(os.path.realpath(__file__)))
from util_functions import *
from myfunctions import *

parser = argparse.ArgumentParser(description='Gene Regulatory Graph Neural Network in ensemble')
# Data from http://dreamchallenges.org/project/dream-5-network-inference-challenge/
# data1: In silico
# data3: E.coli
# data4: Yeast /S. cerevisiae
# Usage:
# python Main_inductive_ensemble.py --traindata-name data3_23 --testdata-name data3_1
# general settings
parser.add_argument('--traindata-name', default='data3'
                                                ,help='train network name')
parser.add_argument('--traindata-name2', default=None, help='also train another network')
parser.add_argument('--testdata-name', default='data3', help='test network name')
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--e', type=float, default=0.05,
                    help='Gaussian noise error')
parser.add_argument('--n', type=int, default=1,
                    help='Run number')

parser.add_argument('--seed', type=int, default=43, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--training-ratio', type=float, default=1.0,
                    help='ratio of used training set')
parser.add_argument('--neighbors-ratio', type=float, default=1.0,
                    help='ratio of neighbors used')
parser.add_argument('--nonezerolabel-flag', default=False,
                    help='whether only use nonezerolabel flag')
parser.add_argument('--nonzerolabel-ratio', type=float, default=1.0,
                    help='ratio for nonzero label for training')
parser.add_argument('--zerolabel-ratio', type=float, default=0.0,
                    help='ratio for zero label for training')
# For debug
parser.add_argument('--feature-num', type=int, default=3,
                    help='feature num for debug')
# Pearson correlation
parser.add_argument('--embedding-dim', type=int, default=1,
                    help='embedding dimmension')
parser.add_argument('--pearson_net', type=float, default=0.8, #1
                    help='pearson correlation as the network')
parser.add_argument('--mutual_net', type=int, default=3, #3
                    help='mutual information as the network')
# model settings
parser.add_argument('--hop', type=int, default=1,
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None,
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=True,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  #use_cuda
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

#random.seed(cmd_args.seed)
#np.random.seed(cmd_args.seed)
#torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


if args.traindata_name is not None:
    # Select data(ori) name
    trdata_name = args.traindata_name.split('_')[0]
    tedata_name = args.testdata_name.split('_')[0]

    #Prepare data(ori)
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))

    trainNet_ori = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.csc'.format(args.traindata_name)),allow_pickle=True)
    trainGroup = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.allx'.format(trdata_name)),allow_pickle=True)
    allxt = trainGroup.toarray().astype('float32')
    dim = len(np.where(allxt[0, :] != 0)[0])
    allxt = allxt[:,:dim]
 #   print('allxt:\n',allxt)


  #  allxt1=add_gauss_noise(allxt, μ=0, sigma=(args.e/3))
    error = np.random.normal(0, (args.e/3),allxt.shape)
    allxt1 = allxt + error


    file_dir = cur_dir + '/' + trdata_name+ '/dream/'
    file_name=trdata_name+'_e=' + str(args.e)+'_' + str(args.n)+'.npy'
    print('file_dir:',cur_dir)
    print('file_name:',file_name)
    print(args.traindata_name)
    np.save(file_dir + file_name, allxt1)
   # print('allxt1:\n',allxt1)



    '''
    pca = PCA(n_components=2)
    pcaAttr = pca.fit_transform( allxt)
    pcaAttr1 = pca.fit_transform(allxt1)
    noise_visual(pcaAttr, pcaAttr1, sigma=(args.error/3),filename=file_name)
    '''






