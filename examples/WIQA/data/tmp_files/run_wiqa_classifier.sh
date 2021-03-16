#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py --dataset_basedir ../../../../data/WIQA/ \
                         --lr 2e-5  --max_epochs 20 \
