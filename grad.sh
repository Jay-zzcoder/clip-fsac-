#!/bin/bash

few_shots=(0)


for few_num in "${!few_shots[@]}";do
## train on the VisA dataset
    base_dir=winclip_mvtec
    save_dir=./exps_${base_dir}/mvtecvit_large_14_518/

    CUDA_VISIBLE_DEVICES=0 python grad.py --dataset visa \
    --data_path /data/sys/data/visa-fewshot/VISA-1cls/ --save_path ./results/mvtec_${base_dir}/few_shot_${few_shots[few_num]} \
    --model ViT-B-16-plus-240 --pretrained openai --image_size 240 
    wait
done



