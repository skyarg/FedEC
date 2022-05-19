#!/bin/bash

python main.py --dataset cifar100 --num_users 100 --num_classes 100 --alg fedec --model cnn --shard_per_user 5 --frac 0.1 --local_bs 10 --lr 0.01 --epochs 200 --local_ep 15 --local_rep_ep 1 --gpu 0 --seed 300 --is_decay 1 --KL_label_weight 0.8 --KL_feature_weight 1.0 --l2_weight 0.05 --l2_type others --max_T 20 --min_T 2

