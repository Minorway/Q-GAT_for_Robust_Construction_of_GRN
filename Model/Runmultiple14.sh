#!/bin/bash

#for((i=1;i<=3;i=i+1))
for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.05 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.1 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.15 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.2 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.25 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.3 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_MemPooling.py --e 0.35 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.4 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.45 --n ${i} --traindata-name 'data4'
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_SAGPool.py --e 0.5 --n ${i} --traindata-name 'data4'
done