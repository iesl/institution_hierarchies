#!/usr/bin/env bash

set -exu

config_dir=$1

$PYTHON_EXEC -m main.train.continue_train_model -c $config_dir 
