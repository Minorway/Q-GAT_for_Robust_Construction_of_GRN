# Quadratic Graph Attention Network (Q-GAT) for Robust Construction of Gene Regulatory Networks
This repository provides scripts to reproduce the results in the paper "Quadratic Graph Attention Network (Q-GAT) for Robust Construction of Gene Regulatory Networks", including model training and SNR calculation, etc. In this work,

1. We propose Q-GAT for the task of GRNs, which is the pioneer in addressing the stability issue of constructing GRNs. Instead of a modification for network structures, our innovation is at the neuronal level from the perspective of neuronal diversity.
2. Systematic experiments show that under different levels of adversarial noises, the proposed Q-GAT is superior to its competitors in terms of stability in constructing GRNs.
3. Not just satisfied with the superior performance ofQ-GAT, we move one step further to analyze why Q-GAT is more robust aided by the SNR and interpretability analyses.

![Q-GAT](https://github.com/Minorway/Q-GAT_for_Robust_Construction_of_GRN/blob/main/Images/Q-GAT_structure.pdf)



We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  



## Repository organization

### Requirements
Some Python packages should be installed before running the scripts, including

* torch                       1.8.0
* torch-geometric             2.1.0
* torch-scatter               2.0.6
* torch-sparse                0.6.9
* torchaudio                  0.8.0
* torchvision                 0.9.0
* tqdm                        4.64.1
 
### Organization
```
Q-GAT_for_Robust_Construction_of_GRN
└─  Model
     │   Main_Q-GAT.py # Train for Q-GAT
     │   graph_classifiers # Model of Q-GAT
└─  SNR
     │   NNs_QNN_comparision.ipynb # NNs and QNN
     │   test # SNR data
└─  agent model
     │   data3_fea=4_Q_GAT0.pt # Pre traing model in $E. coli$ dataset
     │   data4_fea=4_Q_GAT0.pt # Pre traing model in $S. cerevisiae$ dataset
```

### Datasets
We use the public $E. coli$ and $S. cerevisiae$ datasets from the DREAM5 challenge [1] in our article. 

## Contact
If you have any questions about our work, please contact the following email address:

21S012063@stu.hit.edu.cn

Enjoy your coding!



## Reference
[1] Daniel Marbach, James C Costello, Robert Kuffner, Nicole MVega, Robert J Prill, Diogo M Camacho, Kyle R Allison, Manolis Kellis, James J Collins, and Gustavo Stolovitzky. Wisdomof crowds for robust gene network inference. Nature Methods,9(8):796–804, 2012.

