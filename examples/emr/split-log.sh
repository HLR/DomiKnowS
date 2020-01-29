#!/bin/bash

csplit -s -z -n 4 -f "$1". $1 "/^Start for phrase \[.*\]$/" {*}

