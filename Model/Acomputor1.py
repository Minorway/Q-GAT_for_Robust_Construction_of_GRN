import argparse
import os.path
cur_dir = os.path.dirname(os.path.realpath(__file__))
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindata-name', default='data4', help='train network name')
    parser.add_argument('--e', type=float, default=0.0, help='Gaussian noise error')
    parser.add_argument('--n', type=int, default=1, help='Run number')
    parser.add_argument('--feature-num', type=int, default=4,help='feature num for debug')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
    trdata_name = args.traindata_name

    acc_path = cur_dir + '/acc_result_'+args.traindata_name+'/'
    data_path = acc_path+'all_acc_data/'


   # e_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    e_list = [0.0,0.05, 0.1, 0.15, 0.2, 0.25,0.3]
    model_list = ['DGCNN','GAT','DiffPool','GMT','PiNet','ARMA','DeepGCN','SAGPool','TopKPool','GraphSAGE','GQT_v21']


    for model_name in model_list:

        print('\n')
        print('start:',model_name)

        means_list = []
        stds_list = []
        all_data = []

        for i in e_list:
            file = data_path+trdata_name+'_'+model_name+'_e='+str(i)+'.txt'

            result = np.loadtxt(file, dtype=float, delimiter=None, unpack=False)

            mean = result.mean(0)
            mean=np.around(mean, 3)
            std = result.std(0)
            std = np.around(std, 3)

            means_list.append(mean[0])
            stds_list.append(std[0])

        print('means_list：',means_list)
        print('stds_list:',stds_list)

        for i in range(len(means_list)):
            final_result = str(means_list[i])+'±'+str(stds_list[i])
            print(final_result)

            all_data.append(final_result)

            # Output results
            with open(acc_path+'final_result'+'/'+trdata_name+'_'+model_name+'.txt', 'a+') as f:
                f.write(final_result+'\t')

            # Output average moise acc
            with open(acc_path+'acc_noise'+'/'+trdata_name+'_'+model_name+'.txt', 'a+') as f:
                f.write(str(means_list[i])+'\t')


        for j in range(len(all_data)):
            with open(acc_path + 'final_result' + '/' + trdata_name + '_' + 'Allmodel_data' + '.txt', 'a+') as f:
                if j == len(all_data)-1 :
                    f.write(all_data[j] + '\n')
                else:
                    f.write(all_data[j] + '\t')








