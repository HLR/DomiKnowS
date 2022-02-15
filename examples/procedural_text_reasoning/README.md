### Packages should be installed:



pip install networkx
pip install transformers
pip install torch
pip install wget
pip install numpy
pip install gitdb
pip install graphviz
pip install owlready2
pip install ordered_set
pip install pandas
pip install gurobipy

### run the code

CUDA_VISIBLE_DEVICES=6 python WIQA_aug.py --cuda 3 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --pd True



git log

vim翻页
ctrl+f
ctrl+b

rollback 到某个版本
git reset --hard 3ca4a0308f5d098478f9f4310026563dd17a5755

git reset --hard origin

git从指定分支更新代码到本地:

import sys
sys.path.append('../../')

