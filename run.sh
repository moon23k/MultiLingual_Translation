#!/bin/bash

while getopts i:p:t: flag; do
    case "${flag}" in
        a) action=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

if [ $action='train' ] && [ $model ='generator' ]; then
    python3 gen_train.py

if [ $action='train' ] && [ $model ='discriminator' ]; then
    python3 dis_train.py

if [ $action='train' ] && [ $model ='seqGAN' ]; then
    python3 train.py