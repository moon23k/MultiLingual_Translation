#!/bin/bash

python3 -m pip install -U PyYAML
python3 -m pip install sentence piece


while getopts i:p:t: flag; do
    case "${flag}" in
        a) action=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

if [ $action='train' ] && [ $model ='generator' ]; then
    python3 gen_train.py

elif [ $action='train' ] && [ $model ='discriminator' ]; then
    python3 dis_train.py

elif [ $action='train' ] && [ $model ='seqGAN' ]; then
    python3 train.py

elif [ $action='generate' ]; then
    python3 generate_samples.py
fi