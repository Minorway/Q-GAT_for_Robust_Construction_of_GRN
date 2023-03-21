from typing import List
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
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from inspect import signature
from software.pytorch_DGCNN.util import cmd_args
sys.path.append('%s/software/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/software/node2vec/src' % os.path.dirname(os.path.realpath(__file__)))
sys.path.append('%s/graph_classifiers/' % os.path.dirname(os.path.realpath(__file__)))
cur_dir = os.path.dirname(os.path.realpath(__file__))
print("cur_dir",cur_dir)
from util_functions import *
from myfunctions import *
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch import nn
import torch.nn.functional as F
from Mygnns import *
from tqdm import tqdm
from TopKPool import *


parser = argparse.ArgumentParser(description='Gene Regulatory Graph Neural Network in ensemble')
# Data from http://dreamchallenges.org/project/dream-5-network-inference-challenge/
# data1: In silico
# data3: E.coli
# data4: Yeast /S. cerevisiae
# Usage:
# python Main_inductive_ensemble.py --traindata-name data3_23 --testdata-name data3_1
# general settings
parser.add_argument('--traindata-name', default='data4', help='train network name')
parser.add_argument('--traindata-name2', default=None, help='also train another network')
parser.add_argument('--testdata-name', default='data4', help='test network name')
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--e', type=float, default=0.1,
                    help='Gaussian noise error')
parser.add_argument('--n', type=int, default=1,
                    help='Run number')
parser.add_argument('--noisedata', default=False,
                    help='Test noise data(ori)')

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
parser.add_argument('--feature-num', type=int, default=4,
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
#args = parser.parse_args()
args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()  #use_cuda
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)
print('--traindata-name',args.traindata_name)
print('--testdata-name',args.testdata_name)
print('--noisedata',args.noisedata)

random.seed(cmd_args.seed) #cmd_args.seed=1
np.random.seed(cmd_args.seed)  #cmd_args.seed=1
torch.manual_seed(cmd_args.seed) #cmd_args.seed=1
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data(ori)'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
#print("args.file_dir:",args.file_dir)
#print(os.path.realpath('__file__'))
#print(args.file_dir)

# data1: top 195 are TF
# data3: top 334 are TF
# data4: top 333 are TF
# Human: top 745 are TF
dreamTFdict={}
dreamTFdict['data1']=195
dreamTFdict['data3']=334
dreamTFdict['data4']=333
dreamTFdict['Human']=745

# Inductive learning
# Training on 1 data(ori), test on 1 data(ori)
if args.traindata_name is not None:
    # Select data(ori) name
    trdata_name = args.traindata_name.split('_')[0]
    tedata_name = args.testdata_name.split('_')[0]

    #Prepare Training
    trainNet_ori = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.csc'.format(args.traindata_name)),allow_pickle=True)
    trainGroup = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.allx'.format(trdata_name)),allow_pickle=True)
    # Pearson's correlation/Mutual Information as the starting skeletons
    trainNet_agent0 = np.load(args.file_dir+'/data(ori)/dream/'+trdata_name+'_pmatrix_'+str(args.pearson_net)+'.npy',allow_pickle=True).tolist()
    trainNet_agent1 = np.load(args.file_dir+'/data(ori)/dream/'+trdata_name+'_mmatrix_'+str(args.mutual_net)+'.npy',allow_pickle=True).tolist()
    # Random network as the starting skeletons
    allx =trainGroup.toarray().astype('float32')
    # Debug: choose appropriate features in debug
    #trainAttributes = genenet_attribute_feature(allx, dreamTFdict[trdata_name], args.feature_num)

    train_pos, train_neg, test_pos, test_neg = sample_neg_TF(trainNet_ori, 0.3, TF_num=dreamTFdict[trdata_name],max_train_num=args.max_train_num)
    use_pos_size = math.floor(len(train_pos[0]) * args.training_ratio)
    use_neg_size = math.floor(len(train_neg[0]) * args.training_ratio)
    train_pos = (train_pos[0][:use_pos_size], train_pos[1][:use_pos_size])
    train_neg = (train_neg[0][:use_neg_size], train_neg[1][:use_neg_size])
    #trainAttributes = genenet_attribute(allx,dreamTFdict[trdata_name])
    # Debug: choose appropriate features in debug
    trainAttributes = genenet_attribute_feature(allx,dreamTFdict[trdata_name],args.feature_num)

    #Prepare Testing
   #testNet_ori = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.csc'.format(args.testdata_name)),allow_pickle=True)
    #原始数据
    if args.noisedata==False:
        #原始数据
        testGroup = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.allx'.format(tedata_name)),allow_pickle=True)
        #Pearson's correlation/Mutual Information as the starting skeletons
        testNet_agent0 = np.load(args.file_dir+'/data(ori)/dream/'+tedata_name+'_pmatrix_'+str(args.pearson_net)+'.npy',allow_pickle=True).tolist()
        testNet_agent1 = np.load(args.file_dir+'/data(ori)/dream/'+tedata_name+'_mmatrix_'+str(args.mutual_net)+'.npy',allow_pickle=True).tolist()
        allxt = testGroup.toarray().astype('float32')
        testAttributes = genenet_attribute_feature(allxt, dreamTFdict[tedata_name], args.feature_num)
    else:
        #噪声数据
        feature_file_path = cur_dir + '/'+trdata_name+'/dream/'
        N_data = tedata_name +'_e=' + str(args.e)+'_'+str(args.n)+'.npy'
        P_name = tedata_name +'_e=' + str(args.e) + '_' + str(args.n) + '_pmatrix_' + str(args.pearson_net) + '.npy'
        #M_name = tedata_name +'_e=' + str(args.error) + '_' + str(args.n) + '_mmatrix_' + str(args.mutual_net) + '.npy'

        #testNet_ori = np.load(os.path.join(args.file_dir, 'data(ori)/dream/ind.{}.csc'.format(args.testdata_name)),allow_pickle=True)
        testGroup = np.load(feature_file_path+N_data,allow_pickle=True)
        #Pearson's correlation/Mutual Information as the starting skeletons
        testNet_agent0 = np.load(feature_file_path+P_name,allow_pickle=True).tolist()
        #testNet_agent1 = np.load(feature_file_path+M_name,allow_pickle=True).tolist()
        allxt =testGroup.astype('float32')
        testAttributes = genenet_attribute_feature(allxt,dreamTFdict[tedata_name],args.feature_num)


'''Train and apply classifier'''
Atrain_agent0 = trainNet_agent0.copy()  # the observed network
Atrain_agent1 = trainNet_agent1.copy()
Atest_agent0 = testNet_agent0.copy()  # the observed network
#Atest_agent1 = testNet_agent1.copy()
Atest_agent0[test_pos[0], test_pos[1]] = 0  # mask test links
Atest_agent0[test_pos[1], test_pos[0]] = 0  # mask test links
#Atest_agent1[test_pos[0], test_pos[1]] = 0  # mask test links
#Atest_agent1[test_pos[1], test_pos[0]] = 0  # mask test links

# train_node_information = None
# test_node_information = None
if args.use_embedding:

    train_embeddings_agent0 = generate_node2vec_embeddings(Atrain_agent0, args.embedding_dim, True, train_neg)
    train_node_information_agent0 = train_embeddings_agent0
    print("embedding1")
    test_embeddings_agent0 = generate_node2vec_embeddings(Atest_agent0, args.embedding_dim, True, test_neg)
    test_node_information_agent0 = test_embeddings_agent0
    print("embedding2")

    '''
    train_embeddings_agent1 = generate_node2vec_embeddings(Atrain_agent1, args.embedding_dim, True, train_neg) #?
    train_node_information_agent1 = train_embeddings_agent1
    print("embedding3")
    test_embeddings_agent1 = generate_node2vec_embeddings(Atest_agent1, args.embedding_dim, True, test_neg) #?
    test_embeddings_agent1_2d = generate_node2vec_embeddings(Atest_agent1, 32, True, test_neg)
    test_node_information_agent1 = test_embeddings_agent1
    print("embedding4")
    '''

if args.use_attribute and trainAttributes is not None:
    if args.use_embedding:
        train_node_information_agent0 = np.concatenate([train_node_information_agent0, trainAttributes], axis=1)
        test_node_information_agent0 = np.concatenate([test_node_information_agent0, testAttributes], axis=1)

        #train_node_information_agent1 = np.concatenate([train_node_information_agent1, trainAttributes], axis=1)
        #test_node_information_agent1 = np.concatenate([test_node_information_agent1, testAttributes], axis=1)
    else:
        train_node_information_agent0 = trainAttributes
        test_node_information_agent0 = testAttributes
        train_node_information_agent1 = trainAttributes
        test_node_information_agent1 =  testAttributes


#Original
train_graphs_agent0, test_graphs_agent0, max_n_label_agent0 = extractLinks2subgraphs(Atrain_agent0, Atest_agent0, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information_agent0, test_node_information_agent0)
#train_graphs_agent1, test_graphs_agent1, max_n_label_agent1 = extractLinks2subgraphs(Atrain_agent1, Atest_agent1, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, train_node_information_agent1, test_node_information_agent1)
# For training on 2 datasets, test on 1 dataset
#print('# train: %d, # test: %d' % (len(train_graphs_agent0), len(test_graphs_agent0)))

train_batch0,test_batch0=Graph_conversion(train_graphs_agent0,test_graphs_agent0)
#train_batch1,test_batch1=Graph_conversion(train_graphs_agent1,test_graphs_agent1)


def loop_train(data, model, loss_fn, optimizer,bsize=50):
    n_samples = len(data)
    sample_idxes=list(range(len(data)))
    total_iters = (len(data) + (bsize - 1)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    total_loss=0

    model.train()
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]
        graphs=[data[idx] for idx in selected_idx]

        X = Graph_conversion1(graphs).to(device)
        y = X.y.type(torch.long).to(device)

        # Compute prediction and loss
        pred ,_= model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_tem=loss.detach().item()
        total_loss+=loss_tem

    avg_loss = total_loss/ n_samples

    print('\033[92mtrain avg_loss:%.5f\033[0m'% avg_loss)
    return avg_loss

def loop_test(data, model, loss_fn,bsize=50):
    size = len(data)
    sample_idxes = list(range(len(data)))
    total_iters = (len(data) + (bsize - 1)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    test_loss, correct = 0, 0
    y_scores= []
    pre_label = []
    true_label= []
    features = []

    model.eval()
    with torch.no_grad():
        for pos in pbar:
            selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]
            graphs = [data[idx] for idx in selected_idx]
            X = Graph_conversion1(graphs).to(device)
            y = X.y.type(torch.long).to(device)

            pred, lastfeature = model(X)
            y_scores.append(pred)
            features.append(lastfeature)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            pred_labels = (pred.argmax(1)).type(torch.float)
            pre_label.append(pred_labels)
            true_label.append(y)

        correct /= size
        test_loss/=size
    print(f"\033[93mTest result:\nAccuracy: {(100 * correct):>0.1f}%, avg_loss: {test_loss:>8f} \n\033[0m")
    features = torch.cat(features).cpu().detach().numpy()
    y_scores=torch.cat(y_scores).cpu().detach().numpy()
    pre_label = torch.cat(pre_label).cpu().detach().numpy()
    true_label = torch.cat(true_label).cpu().detach().numpy()
    return features,y_scores,pre_label,true_label


def train_main(train_graphs,test_graphs,classifier, loss_fn, optimizer,epochs=50,model_path=''):
    epoch_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss=loop_train(train_graphs,classifier, loss_fn, optimizer,bsize=50)
        epoch_losses.append(epoch_loss)
        _,_,_,_=loop_test(test_graphs, classifier, loss_fn,bsize=50)
    torch.save(classifier.state_dict(), model_path)
    print("Done!")
    plt.title('cross entropy averaged over minibatches')
    plt.plot(epoch_losses)
    #save_path = cur_dir + '/acc_result_data3/ROC_curve/'
    #plt.savefig(save_path + 'epoch_losses_'+str(args.n) + '.jpg',dpi=500,bbox_inches = 'tight')
    #plt.show()


if __name__ == '__main__':
    num = args.feature_num
    model_path = cur_dir + '/agent_model/' + trdata_name + '_fea=' + str(num) + '_TopKPool'+str(0)+'.pt'

    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = Net(num_features=2,num_classes=2).to(device)

    learning_rate = 1e-4
    weight_decay1 = 0.01
    weight_decay2 = 5e-4
    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

   # train_main(train_graphs_agent0, test_graphs_agent0,classifier, loss_fn, optimizer,epochs=100 ,model_path=model_path)

    classifier.load_state_dict(torch.load(model_path))
    features,y_scorest,pre_label,true_label=loop_test(test_graphs_agent0, classifier, loss_fn,bsize=50)

    y_true=true_label.copy()
    y_scores=y_scorest.copy()[:,1]

    # Visualization of classification results
    file_path = cur_dir + '/draw/'+trdata_name+'/'
    if args.noisedata == False and args.n == 1:
        tsne_visual(features, y_true,file_path, filename='TopKPool0')
    if args.noisedata == True and args.e == 0.1:
        tsne_visual(features, y_true,file_path, filename='TopKPool')

    tn, fp, fn, tp = confusion_matrix(y_true, pre_label).ravel()
    accuracy,precision,recall,fpr=evaluation_index(tn,fp,fn,tp)
    auc=roc_auc_score(y_true, y_scores)

    # 绘制ROC曲线
    save_path = cur_dir + '/draw/roc_curve/' + trdata_name + '/'
    if args.noisedata == False:
        draw_ROC_curve(y_true,y_scores,save_path,filename='TopKPool')


    pri=str(tp)+"\t"+str(fp)+"\t"+str(tn)+"\t"+str(fn)+'\n'
    print('tn fp fn tp:\n',pri)
    result=str(accuracy)+"\t"+str(precision)+"\t"+str(recall)+"\t"+str(fpr)+"\t"+str(auc)+'\n'
    print("TopKPool_acc.txt:accuracy precision recall fpr auc\n",result)


    # Output results
    with open(cur_dir+'/acc_result_'+trdata_name+'/'+trdata_name+'_'+'TopKPool'+'_e='+str(args.e)+'.txt', 'a+') as f:
        f.write(result)
