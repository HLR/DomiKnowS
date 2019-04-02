#!/usr/bin/env bash

python=python3.7
apt update
apt install -y $python $python-dev $python-distutils
apt install -y curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$python get-pip.py

$python -m pip install -r requirements.txt
$python -m pip install -U https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl

export PYTHONPATH=$PYTHONPATH:$(cd ../../regr && pwd)
