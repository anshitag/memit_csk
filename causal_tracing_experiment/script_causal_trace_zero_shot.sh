# script for performing causal tracing experiment on zero shot model

# preparing data for causal tracing for zero shot model
# input should be the input json file eg data/20q/edit_set.json
python3 prepare_zero_shot_trace_data.py -m <model_name> -i <input_file> -svo <svo_annotated_file> -o <out_file>


# Running the causal tracing experiment
python3 causal_trace_zero_shot.py -i <out_file> --model_name <gpt2-large | gpt2-xl> --dataset <dataset_name>  --experiment <subject | object | verb>

