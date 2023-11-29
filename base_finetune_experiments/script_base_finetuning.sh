#!/bin/bash

# Example script for running base finetuning on the gpt2-xl model for the 20q datasets

python3 fine_tune_gpt.py \
--model gpt2-xl \
--dataset 20q \
--log2_batch_size 6 \
--epochs 10 \
--learning_rate 1.4325476949817346e-05 \
--save_path ./result/checkpoint/ \
--output_path ./result/output/ \
--train_file ../data/20q/train_set.json \
--valid_file ../data/20q/edit_validation_set.json \
--test_file ../data/20q/edit_set.json \
--save_model
