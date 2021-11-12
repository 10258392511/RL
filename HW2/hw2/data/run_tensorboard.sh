#!/bin/bash

for dirname in $(ls .)
do
    if [[ -d $dirname ]]
    then
        echo "now at ${dirname}"
        tensorboard --logdir $dirname
    fi
done
