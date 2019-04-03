#!/usr/bin/env bash

data_url="http://www.cs.tulane.edu/~pkordjam/data/data.zip"
relative_path="data/EntityMentionRelation"

if [ ! -d "$relative_path" ]; then
    wget "$data_url"
    unzip "$(basename "$data_url")" "$relative_path/*"
else
    echo 'Data already extracted.'
fi
