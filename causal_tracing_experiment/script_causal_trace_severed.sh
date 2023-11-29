# script for performing severed causal tracing experiment

# preparing data for severed causal tracing
# input should be the output file generated after model inference
python3 prepare_trace_data.py -i <out_json_file_after_model_inference> -svo <svo_annotated_file> -o <out_file>



# Running the severed causal tracing experiment
python3 causal_trace_severed.py -i <out_file> --model_name <gpt2-large | gpt2-xl> --dataset <dataset_name>  --experiment <subject | object | verb> \
 --checkpoint <model_checkpoint>

# Getting the average indirect effects and plot them in a graph
python average_causal_effects_severed.py --input_dir <out-dir of previous step> --experiment <subject | object | verb> --model <gpt2-large | gpt2-xl> --fact_file <out_file>