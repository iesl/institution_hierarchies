#!/usr/bin/env bash

set -exu

config_file=$1
fold=$2

$PYTHON_EXEC -m main.eval.test_baseline -c $config_file -f $fold
