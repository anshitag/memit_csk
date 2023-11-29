#!/bin/bash


#Example script for running repair finetuning on the base finetuned gpt2-xl model for the 20q datasets
python fine_tune_gpt.py \
--model gpt2-xl \
--dataset 20q \
--log2_batch_size 5 \
--epochs 9 \
--learning_rate 1.5900193740411122e-06 \
--checkpoint ../base_finetune_experiments/result/checkpoint/ \
--save_path ./result/checkpoint/ \
--output_path ./result/output/ \
--train_file ../data/20q/incorrect_subsets/gpt2-xl/edit_validation_subset.json \
--valid_file ../data/20q/edit_validation_set.json \
--save_model


#Example script for evaluating repair finetuning on repaired gpt2-xl model for the affected metrics for the 20q datasets
python evaluate_affected_finetune_model.py \
--model gpt2-xl \
--model_path ./result/checkpoint/ \
--input_file ../data/20q/probe_set.json


#Example script for evaluating repair finetuning on repaired gpt2-xl model for the unaffected metrics for the 20q datasets
python evaluate_unaffected_finetune_model.py \
--model gpt2-xl \
--edit_model_path ./result/checkpoint/ \
--model_path ../base_finetune_experiments/result/checkpoint/ \
--input_file ../data/20q/probe_set.json
