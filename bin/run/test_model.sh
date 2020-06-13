#!/usr/bin/env bash

set -exu

config_dir=$1
is_parallel=${2-False}
shard=${3-False}

$PYTHON_EXEC -m main.eval.test_model -c $config_dir -p $is_parallel -s $shard
