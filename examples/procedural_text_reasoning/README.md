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


CUDA_VISIBLE_DEVICES=0 python WIQA_aug.py --cuda 0 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --pd True

CUDA_VISIBLE_DEVICES=0 python WIQA_aug.py --cuda 0 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --pd True


python WIQA_aug.py --cuda 0 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --pd True

CUDA_VISIBLE_DEVICES=1 python WIQA_aug.py --cuda 0 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --pd True

CUDA_VISIBLE_DEVICES=1 python WIQA_aug.py --cuda 0 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --semantic_loss True



git log

vim翻页
ctrl+f
ctrl+b

rollback 到某个版本
git reset --hard 3ca4a0308f5d098478f9f4310026563dd17a5755

git reset --hard origin

git从指定分支更新代码到本地:
git pull origin chen_zheng_procedural_text

import sys
sys.path.append('../../')


SampleLossProgram位置：
分支：feature/new_loss_function
位置：regr/program/lossprogram.py
     regr/program/model/lossModel.py

regr/graph/dataNode.py class 重新导入 因为lossmodel报错 # AttributeError: type object 'DataNode' has no attribute 'tnormsDefault'
regr/program/program.py class重新导入， 因为TypeError: # __init__() got an unexpected keyword argument 'sample'


Hossein给的代码：





