#!/usr/bin/env bash

set -exu

inputfile=$1
outputfile=$2
tokenizer=$3
min_count=$4

$PYTHON_EXEC -m entity_align.setup.MakeVocab $inputfile $outputfile $tokenizer $min_count
