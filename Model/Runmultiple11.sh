#!/bin/bash

#for((i=1;i<=3;i=i+1))
for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.05 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.1 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.15 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.2 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.25 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.3 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.35 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.4 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.45 --n ${i}
done


for i in $(seq 10)
do
  echo 'n='${i}
  python --version
  python Main_GCNII.py --e 0.5 --n ${i}
done