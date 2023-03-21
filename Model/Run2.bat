@echo off

python Main_ARMA.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 15 --noisedata True
python Main_DeepGCN.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_DGCNN.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_DiffPool.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_GCNFN.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_GMT.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_GraphSAGE.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_PiNet.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_SAGPool.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True
python Main_TopKPool.py --traindata-name "data3" --testdata-name "data3" --e 0.1 --n 1 --noisedata True


python Main_ARMA.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_DeepGCN.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_DGCNN.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_DiffPool.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_GCNFN.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_GMT.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_GraphSAGE.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_PiNet.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_SAGPool.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True
python Main_TopKPool.py --traindata-name "data4" --testdata-name "data4" --e 0.1 --n 1 --noisedata True