import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy.stats import pearsonr
import scipy.sparse
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 8})
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import scienceplots

def max_min_normalization(allx): #对数据进行归一化处理
    #所有的数据 x 分布在 [0 ， 1] 之间
    allx_n = np.zeros(allx.shape,dtype='float32')
    for i in range(allx.shape[0]):
        max=allx[i].max()
        min=allx[i].min()
        nor=(allx[i]-min)/(max-min)
        allx_n[i]=nor
    return allx_n

def vector_add_gauss_noise(vector,μ=0, sigma=1):
    # 对输入数据加入gauss噪声
    noise=np.zeros(vector.size)
    for i in range(len(vector)):
        noise[i] += random.gauss(μ, sigma)
    noise_vector=vector+noise
    return noise_vector

def data_processing(data,ax=1): # 数据在（0，1）范围之外，调整为该列或行的平均值
    mean=data.mean(axis=ax)
    for i in range(data.shape[0]):
        for j in range(len(data[i])):
            if data[i,j]>1 or data[i,j]<0 :
                data[i,j]=mean[i]
    return data


def Random_labeling(lenth,rate,search=0):
    """
    二分类随机标签
    @param lenth: 数据点的个数
    @param rate: 为0的标签比率
    @param search: 标签随机排序
    @type lenth:int
    @type search:int
    @type rate:float
    @return: 随机标签列表
    """

    """
    trueList=[]
    for i in range(len(test_pos[0])):
    trueList.append(1)
    for i in range(len(test_neg[0])):
    trueList.append(0)
    """

    num_of_0=int(lenth*rate)
    num_of_1=lenth-num_of_0
    label_0 = np.zeros(num_of_0, dtype=int).tolist()
    label_1 = np.ones(num_of_1, dtype=int).tolist()
    label = label_0 + label_1
    if search==0:
        np.random.shuffle(label)
    elif search==1:
        random.shuffle(label)
    else:
        print("trueLabel")
    return label

def Index_position(label):
    """
    获取数据点标签是0和1的位置
    @param label:0和1的标签列表
    @type label:list
    @return:位置列表
    """
    is_0_position = []
    is_1_position = []
    for i in range(len(label)):
        if label[i] == 0:
            is_0_position.append(i)
        else:
            is_1_position.append(i)
    return is_0_position, is_1_position

def Two_dim_data(data,label):
    """
    0和1标签数据分类
    @param data: 有带0、1标签的二维数据
    @param label: 数据标签
    @return: 0和1两类数据矩阵
    """
    pos0, pos1 = Index_position(label)
    X_0 = data[pos0]
    X_1 = data[pos1]
    return X_0,X_1

def draw_scatter(data1_0,data1_1,data2_0,data2_1):
    plt.subplot(121)
    plt.scatter(data1_0[:, 0], data1_0[:, 1], color='r', marker='.')
    plt.scatter(data1_1[:, 0], data1_1[:, 1], color='g', marker='.')
    plt.title("Before classification")
    plt.subplot(122)
    plt.scatter(data2_0[:, 0], data2_0[:, 1], color='r', marker='.')
    plt.scatter(data2_1[:, 0], data2_1[:, 1], color='g', marker='.')
    plt.title("After classification")
    plt.show()

from matplotlib import cm

def plot_with_labels(lowDWeights, labels,textname='123'):
    plt.cla()
    # 降到二维了，分别给X和Y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        if s==0:
            v=int(255/7*2)
        else:
            v=int(255/7*6)
        c = cm.rainbow(v) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.scatter(x, y, c=c, marker='.')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()); plt.title('Visualization of classification results')
    plt.savefig(cur_dir+"/draw/{}.jpg".format(textname),dpi=500,bbox_inches = 'tight')

def plot_with_labels2(test_pos, test_neg, labels,textname='123'):
    plt.cla()
    x=list(test_pos[0])+list(test_neg[0])
    y=list(test_pos[1])+list(test_neg[1])
    X=np.array(x)
    Y=np.array(y)
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        if s==0:
            v=int(255/7*2)
        else:
            v=int(255/7*6)
        c = cm.rainbow(v) # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.scatter(x, y, c=c, marker='.')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()); plt.title('Visualization of classification results')
    plt.savefig(cur_dir+"/draw/{}.jpg".format(textname),dpi=500,bbox_inches ='tight')

def plot_embedding(lowDWeights, labels,file_path,filename="visiual_123"):
    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.cla()
    pos = lowDWeights[np.where(labels == 1)]
    neg =lowDWeights[np.where(labels == 0)]
    posColor = ['#9C0245','#FC7671', '#EB232F']
    negColor = ['#594AA5','#00C0BB', '#168043']
    plt.scatter(pos[:,0], pos[:,1], c=posColor[0], marker='o',s=5,
               edgecolors='white',linewidths=0.1 ,label='pos')
    plt.scatter(neg[:,0], neg[:,1], c=negColor[0], marker='o',s=5,
               edgecolors='white',linewidths=0.1,label='neg')
    # 设置标题
    #plt.title('t-SNE',fontname="Times New Roman", fontsize=20)
    # 设置坐标轴标题
    #plt.xlabel('False positive rate', fontdict={"family": "Times New Roman", "size": 20})
    #plt.ylabel('True positive rate', fontdict={"family": "Times New Roman", "size": 20})
    # 设置x轴的刻标以及对应的标签
    #plt.xlim([-0.03, 1.0])
    #plt.ylim([0.0, 1.03])
    #plt.xticks(fontname="Times New Roman", fontsize=12)
    #plt.yticks(fontname="Times New Roman", fontsize=12)
    plt.xticks([])
    plt.yticks([])
    #设置图例
    plt.legend(loc="upper right", markerscale=2, prop={"family": "Times New Roman", "size": 11})
    plt.savefig(file_path+filename+'.png',dpi=1000,bbox_inches ='tight')


def tsne_visual(data,label,file_path,filename='visual'):

    #x_min, x_max = np.min(data, 0), np.max(data, 0)
    #data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理

    print('Computing t-SNE embedding...')
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    #tsne = TSNE(n_components=2, learning_rate=100)
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)  # TSNE降维，降到2
    X_embedded = tsne.fit_transform(data)

    print('Finished computing.')
    print('draw_plot...')

    plot_embedding(X_embedded, label,file_path,filename)
    print('Finished drawing.')

def load_position(poslist,path):
    for i in range(len(poslist)):
        tem=np.loadtxt(path+poslist[i]).astype('int')
        pos=(tem[0],tem[1])
        if i==0:
           train_pos=pos
        elif i==1:
           train_neg=pos
        elif i==2:
            test_pos=pos
        elif i==3:
            test_neg=pos
    return train_pos,train_neg,test_pos,test_neg

def noise_visual(allx_original,allx_noise,sigma=0.0,filename='file_123'):
    plt.scatter(allx_original[:200, 0], allx_original[:200, 1], marker='.', label='original_data')
    plt.scatter(allx_noise[:200, 0], allx_noise[:200, 1], color='r', marker='.', label='noise_data')
    plt.legend(loc='upper right')
    plt.title("mean={},sigma={}".format(0,sigma))
    plt.savefig(cur_dir+'/data3/visualization/'+filename+'.jpg',dpi=500,bbox_inches = 'tight')

def fpr_tpr_auc(label,scores,pos_label):
    fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr,auc

def ROC_curve(fpr,tpr,auc,filename="roc_curve"):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='roc_curve curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(cur_dir+'/acc_result_data3/ROC_curve/' + filename + '.jpg',dpi=500,bbox_inches = 'tight')
    plt.show()

def Precision_recall(recall,precision,filename="PRC"):
    plt.plot(recall,  precision, color='darkorange',
             lw=2, label='precision-recall curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-recall curve')
    plt.legend(loc="lower right")
    plt.savefig(cur_dir + '/acc_result_data3/ROC_curve/' + filename + '.jpg',dpi=500,bbox_inches = 'tight')



def get_noisy_skeletons(maxrix,tf):
    noise_net = np.zeros((maxrix.shape))
    tf_g=maxrix[0:tf]
    index=np.where(tf_g==1)
    noise_net[index[0],index[1]]=1
    noise_net[index[1],index[0]]=1
    return noise_net

def add_gauss_noise2(allx,μ=0, sigma=0.0333):
    allx_noise = np.zeros(allx.shape)
    for i in range(allx.shape[1]):
        for j in range(len(allx[:,i])):
            e = random.gauss(μ, sigma)
            allx_noise[i,j]=(1+e)*allx[i,j]
    return allx_noise

def add_gauss_noise(allx,μ=0, sigma=0.0333):
    error = np.random.normal(μ, sigma, allx.shape)
    print('error:\n',error)
    allx_noise=(1+error)*allx
    return allx_noise

def evaluation_index(tn,fp,fn,tp):
    accuracy=(tp+tn)/(tn+fp+fn+tp) #ACC = (TP + TN)/(P + N)
    precision=tp/(tp+fp) #PPV = TP/(TP + FP)
    recall=tp/(tp+fn) #TPR = TP/P = TP/(TP + FN)
    fpr=fp/(fp+tn)  #FPR = FP/N = FP/(FP + TN)
    return accuracy,precision,recall,fpr
def std_index(arr,acc):
    arr_sum=np.sum((arr-acc)**2)
    arr_std  = pow(arr_sum/(len(arr)-1),0.5)
    return arr_std


def draw_plot(arr, labels,filename="v_123"):
    plt.cla()
    plt.legend(loc="upper right")
    #plt.title('Visualization of classification results')
    plt.savefig(cur_dir+'/draw/'+filename+'.jpg',dpi=500,bbox_inches = 'tight')

# 绘制ROC曲线
def draw_ROC_curve(y_true,y_score,save_path,filename):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=filename+'(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path+filename+'.jpg',dpi=500,bbox_inches = 'tight')
   # plt.show()

def Graph_object_conversion(graph_list):
    Data_graph=[]
    for GNNGraph in graph_list:
        x = torch.tensor(GNNGraph.node_features)
        edge_index = torch.tensor(GNNGraph.edge_index).type(torch.long)
        label=torch.tensor([GNNGraph.label])
        g = Data(x=x, edge_index=edge_index,y=label)
        Data_graph.append(g)
    return Data_graph

def Graph_conversion(train_graph,test_graph):
    # 图对象类型转换
    train_graphs=Graph_object_conversion(train_graph)
    test_graphs = Graph_object_conversion(test_graph)
    # 打乱图的顺序
    random.shuffle(train_graphs)
    random.shuffle(test_graphs)
    train_batch = Batch.from_data_list(train_graphs)
    test_batch = Batch.from_data_list(test_graphs)
    return train_batch,test_batch

def Graph_conversion1(graph_list):
    # 图对象类型转换
    batch_graphs=Graph_object_conversion(graph_list)
    # 打乱图的顺序
    random.shuffle(batch_graphs)
    batch_graphs = Batch.from_data_list(batch_graphs)
    return batch_graphs

def get_graph_representation(Gmodel,train_batch,test_batch,device):
    train_embed = Gmodel(train_batch.to(device))
    test_embed = Gmodel(test_batch.to(device))
    train_labels = train_batch.y.type(torch.long)
    test_labels = test_batch.y.type(torch.long)
    return train_embed,test_embed,train_labels,test_labels

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    logits = torch.rand(2, 2)
    pred = F.softmax(logits, dim=1)
    pred1 = F.log_softmax(logits, dim=1)
    print(logits)
    print(pred)
    print(pred1)

