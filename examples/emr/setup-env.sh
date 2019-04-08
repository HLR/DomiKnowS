#!/usr/bin/env bash
set -e

# Pre-reqiurement: python 3 and pip module
# It will be good to install pytorch beforehand since there is a bunch of selection

python=python3.7
$python -m pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(cd ../../regr && pwd)
