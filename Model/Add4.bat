@echo off

for /l %%a in (1,1,30) do (
echo n=%%a
python Add_noise4.py --e 0.05 --n %%a
)

for /l %%a in (1,1,30) do (
echo n=%%a
python Add_noise4.py --e 0.1 --n %%a
)

for /l %%a in (1,1,30) do (
echo n=%%a
python Add_noise4.py --e 0.15 --n %%a
)

for /l %%a in (1,1,30) do (
echo n=%%a
python Add_noise4.py --e 0.2 --n %%a
)

for /l %%a in (1,1,30) do (
echo n=%%a
python Add_noise4.py --e 0.25 --n %%a
)

for /l %%a in (1,1,30) do (
echo n=%%a
python Add_noise4.py --e 0.3 --n %%a
)


