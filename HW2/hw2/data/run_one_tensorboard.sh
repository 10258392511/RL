#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo "only need one argument"
    exit 1
fi

source activate cs285
tensorboard --logdir $1
