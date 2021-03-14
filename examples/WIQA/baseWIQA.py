
import wget
import os
import torch
import numpy as np
from regr.program import POIProgram
import subprocess
import sys
import shutil
from torch import nn
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric
from regr.program.loss import NBCrossEntropyLoss

from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, TorchEdgeSensor, JointSensor, FunctionalSensor, \
    FunctionalReaderSensor

print('Downloading dataset')

url = "https://ai2-public-datasets.s3.amazonaws.com/wiqa/wiqa-dataset-v2-october-2019.zip"

if not os.path.exists('data/WIQA/'):
    os.makedirs("data/WIQA")

if not os.path.exists('data/WIQA/wiqa-dataset-v2-october-2019.zip'):
    wget.download(url, 'data/WIQA/wiqa-dataset-v2-october-2019.zip')
    import zipfile
    with zipfile.ZipFile('data/WIQA/wiqa-dataset-v2-october-2019.zip', 'r') as zip_ref:
        zip_ref.extractall('data/WIQA/') # train.jsonl dev.jsonl test.jsonl

if not os.path.exists('data/WIQA/repo'):
    os.makedirs("data/WIQA/repo")
    import git #pip install GitPython
    git.Git("data/WIQA/repo").clone("https://github.com/allenai/wiqa-dataset.git")

    with open("data/WIQA/repo/wiqa-dataset/requirements.txt","r") as fp:
        for package in fp.readlines():
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    os.rename("data/WIQA/repo/wiqa-dataset","data/WIQA/repo/wiqadataset")

if not os.path.exists('data/WIQA_AUG/repo'):
    os.makedirs("data/WIQA_AUG/repo")
    import git #pip install GitPython
    git.Git("data/WIQA_AUG/repo").clone("https://github.com/AkariAsai/logic_guided_qa.git")

    with open("data/WIQA_AUG/repo/logic_guided_qa/requirements.txt","r") as fp:
        for package in fp.readlines():
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])



print(os.getcwd())
sys.path.append("data/WIQA/repo/")
sys.path.append("data/WIQA/repo/wiqadataset/")
sys.path.append("data/WIQA/repo/wiqadataset/model/")

from src.wiqa_wrapper import WIQADataPoint

wimd = WIQADataPoint.get_default_whatif_metadata()
sg = wimd.get_graph_for_id(graph_id="13")
print(sg.to_json_v1())

os.remove("data/WIQA/repo/wiqadataset/model/run_wiqa_classifier.sh")
shutil.copyfile("data/tmp_files/run_wiqa_classifier.sh","data/WIQA/repo/wiqadataset/model/run_wiqa_classifier.sh")

os.remove("data/WIQA/repo/wiqadataset/model/run.py")
shutil.copyfile("data/tmp_files/run.py","data/WIQA/repo/wiqadataset/model/run.py")

#os.chdir("data/WIQA/repo/wiqadataset/")
#subprocess.call(['sh', 'model/run_wiqa_classifier.sh'])

#python wiqa_augmentation.py --data_dir PATH_TO_WIQA_DATA_DIR --output_dir PATH_TO_AUGMENTED_DATA
#subprocess.check_call([sys.executable, 'data/WIQA_AUG/repo/logic_guided_qa/wiqa_augmentation.py',
#                       '--data_dir',"data/WIQA/",'--output_dir',"data/WIQA_AUG/"])
