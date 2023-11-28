#!/bin/bash

#Example script for executing memit_csk on the base finetuned gpt2-xl model for the 20q datasets

#command for hyper parameter tuning on Edit Validation Set for evaluation of configuration generalization
python3 -m memit_csk_experiment.evaluate_csk \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--edit_file=data/20q/incorrect_subsets/gpt2-xl/edit_validation_subset.json \
--inference_type=config \
--inference_file=data/20q/edit_validation_set.json \
--edit_token=object \
--edit_location=last \
--layer_size=3 \
--max_layer=3 \
--v_lr=0.02689498084872511 \
--model_checkpoint=<Base-finetuned model checkpoint> \
--wandb_active

#command for running evalution based on best hyperparamter on Edit Set for evaluation of configuration generalization
python3 -m memit_csk_experiment.evaluate_csk \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--edit_file=data/20q/incorrect_subsets/gpt2-xl/edit_subset.json \
--inference_type=config \
--inference_file=data/20q/edit_set.json \
--edit_token=object \
--edit_location=last \
--layer_size=3 \
--max_layer=3 \
--v_lr=0.02689498084872511 \
--model_checkpoint=<Base-finetuned model checkpoint>


#command for running evalution based on best hyperparamter on Probe Set for evaluation of semantic generalization
python3 -m memit_csk_experiment.evaluate_csk \
--model_name=gpt2-xl \
--hparams_fname=gpt2-xl.json \
--edit_file=data/20q/incorrect_subsets/probe_set.json \
--inference_type=semantic \
--inference_file=data/20q/probe_set.json \
--edit_token=object \
--edit_location=last \
--layer_size=3 \
--max_layer=3 \
--v_lr=0.02689498084872511 \
--model_checkpoint=<Base-finetuned model checkpoint>
