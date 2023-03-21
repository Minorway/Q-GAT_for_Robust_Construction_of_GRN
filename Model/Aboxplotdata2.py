import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
cur_dir = os.path.dirname(os.path.realpath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata-name', default='data3', help='train network name')
    parser.add_argument('--e', type=float, default=0.05, help='Gaussian noise error')
    parser.add_argument('--n', type=int, default=1, help='Run number')
    parser.add_argument('--feature-num', type=int, default=4,help='feature num for debug')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    trdata_name = args.traindata_name

    acc_path = cur_dir + '/acc_result_'+trdata_name+'/'
    data_path = acc_path+'all_acc_data/'
    boxdata_path = acc_path+'boxplot/'


   # e_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    e_list = [0.0,0.05, 0.1, 0.15, 0.2, 0.25,0.3]
    model_list = ['DGCNN','GAT','DiffPool','GMT','PiNet','ARMA','DeepGCN','SAGPool','TopKPool','GraphSAGE']


    for i in range(len(e_list)):
        boxplotdata = np.zeros((10, len(model_list)))
       # boxplotdata = np.zeros((10, len(model_list)))

        boxplot_x = []
        for j in range(len(model_list)):
            file = data_path + trdata_name + '_' + model_list[j] + '_e=' + str(e_list[i]) + '.txt'
            result = np.loadtxt(file, dtype=float, delimiter=None, unpack=False)

            x = np.round(result[:, 0], 5)
            boxplot_x.append(x)

           # boxplotdata[:,j] = np.round(result[:,0],5)
        plt.boxplot(boxplot_x,
                    showfliers=False,
                    patch_artist=True, sym='o',
                    labels=model_list,  # 添加具体的标签名称
                    showmeans=False,
                    boxprops={'color': 'black', 'facecolor': 'White'},  # '#9999ff'  #'White'
                    flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
                    meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'color': 'y', },
                    medianprops={'linestyle': '--', 'color': 'orange'})
        plt.xticks(fontproperties='Times New Roman', fontsize=7)

        #np.savetxt(boxdata_path+trdata_name+'_boxplotdata_e='+str(e_list[i])+'.txt', boxplotdata,fmt='%.18e')
        plt.savefig(cur_dir + '/draw/' + trdata_name + '_boxplots_e = ' + str(e_list[i]) + '.jpg', dpi=500,
                    bbox_inches='tight')

        plt.show()
       # print(boxplotdata)








