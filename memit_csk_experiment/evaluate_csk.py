import json, os
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from dsets import (
    CommonSenseDataset
)
from memit_csk_experiment.py.eval_utils_csk import compute_rewrite_quality_csk, evaluate_model, compare_models, compare_unaffected_output
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT_CSK": (MEMITHyperParams, apply_memit_to_model)
}

DS_DICT = {
    "csk": (CommonSenseDataset, compute_rewrite_quality_csk)
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    model_checkpoint: str,
    hparams_fname: str,
    ds_name: str,
    continue_from_run: str,
    conserve_memory: bool,
    dir_name: str,
    edit_token: str,
    edit_location : str,
    edit_file: str,
    inference_file: str,
    inference_type: str,
    max_layer: int,
    layer_size: int,
    v_lr: float,
    v_num_grad_steps: int,
    kl_factor: float,
    k_sample_size: int = None,
    max_prob: float = None,
    wandb_active: bool = False,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )

    hparams = params_class.from_json(params_path)
    if edit_token is None:
        edit_token = hparams.fact_token.split("_")[0]
    if edit_location is not None and edit_token is not None:
        hparams.fact_token = f"{edit_token}_{edit_location}"
    if max_layer is not None:
        min_layer = max_layer-(layer_size-1) if max_layer-(layer_size-1)>0 else 1
        hparams.layers = list(range(min_layer, max_layer+1))
    if v_lr is not None:
        hparams.v_lr = v_lr
    if v_num_grad_steps is not None:
        hparams.v_num_grad_steps = v_num_grad_steps
    if kl_factor is not None:
        hparams.kl_factor = kl_factor
    if max_prob is not None:
        hparams.max_prob = max_prob

    if not (run_dir / "params.json").exists():
        with open(run_dir / "params.json", 'w') as f:
            json.dump(dict(vars(hparams)), f, indent=4)

    if wandb_active:
        wandb.config.update(dict(vars(hparams)), allow_val_change=True)

    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        if model_checkpoint:
            model = AutoModelForCausalLM.from_pretrained(model_checkpoint).cuda()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(edit_file, tok=tok, noise_token=edit_token, k_sample_size=k_sample_size)

    if inference_file is not None:
        compare_metrics = {}
        sample_indexes = ds.getindexes()
        unedited_eval_metrics, unedited_output_data = evaluate_model(model, tok, inference_file, "unedited", inference_type, sample_indexes)
        compare_metrics.update(unedited_eval_metrics)
        if wandb_active:
            wandb.log(unedited_eval_metrics)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}_{edit_token}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()
        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            noise_token = edit_token,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)

        if SAVE_MODEL:
            edited_model.save_pretrained(str(run_dir)+"/edited_model")
            
        # Evaluate new model
        if inference_file is not None:
            edited_eval_metrics, edited_output_data = evaluate_model(edited_model, tok, inference_file, "edited", inference_type, sample_indexes)
            compare_metrics.update(edited_eval_metrics)
            
            if inference_type=="config":
                compare_metrics.update(compare_models(unedited_output_data, edited_output_data))
                compare_metrics["f1_difference"] = edited_eval_metrics["edited_f1Score"] - unedited_eval_metrics["unedited_f1Score"]
            else:
                compare_metrics.update(compare_unaffected_output(unedited_output_data, edited_output_data))
            if wandb_active:
                wandb.log(compare_metrics)

            out_file = str(run_dir)+"/inference_result_summary.json"
            print(f"Output metrics saved at {out_file}")
            with open(out_file, "w") as f:
                json.dump(compare_metrics, f, indent=1)

            if k_sample_size:
                return compare_metrics

        print("Evaluation took", time() - start)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT_CSK"],
        default="MEMIT_CSK",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run."
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--model_checkpoint",
        default=None,
        help="Fine tuned model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["csk"],
        default="csk",
        help="Dataset to perform evaluations on. Set to csk by default.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1000,
        help="Number of corrections to perform simultaneously. Set to 1000 by default.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--edit_token",
        choices=["subject","verb","object"],
        default="subject",
        help="Token position used to edit model",
    )
    parser.add_argument(
        "--edit_location",
        choices=["first","last"],
        default="last",
        help="Location of the edit token used to edit model",
    )
    parser.add_argument(
        "--save_model",
        dest="save_model",
        action="store_true",
        help="Save edited model, False by default.",
    )
    parser.add_argument(
        "--edit_file",
        type=str,
        default=None,
        help="The commonsense data file to edit model",
        required=True,
    )
    parser.add_argument(
        "--inference_file",
        type=str,
        default=None,
        help="The commonsense data file to compare updates by edited model",
    )
    parser.add_argument(
        "--inference_type",
        choices=["config","semantic"],
        default="config",
        help="The commonsense inference type to measure editing performance",
    )
    parser.add_argument(
        "--max_layer",
        type=int,
        default=None,
        help="Maximum value of layer to edit",
    )
    parser.add_argument(
        "--layer_size",
        type=int,
        default=5,
        help="Size of layers to edit from the max_layer",
    )
    parser.add_argument(
        "--v_lr",
        type=float,
        default=None,
        help="Editing learning rate",
    )
    parser.add_argument(
        "--v_num_grad_steps",
        type=int,
        default=None,
        help="Editing number of gradient steps",
    )
    parser.add_argument(
        "--kl_factor",
        type=float,
        default=None,
        help="Editing KL factor",
    )
    parser.add_argument(
        "--max_prob",
        type=float,
        default=None,
        help="Editing max probability to cutoff during training for target z_s",
    )
    parser.add_argument('--wandb_account', type=str, default="")
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--wandb_active', action='store_true')

    args = parser.parse_args()
    set_seed()

    if args.wandb_active:
        wandb.init(project=args.wandb_project, entity=args.wandb_account, config=args)

    SAVE_MODEL = args.save_model

    main(
        args.alg_name,
        args.model_name,
        args.model_checkpoint,
        args.hparams_fname,
        args.ds_name,
        args.continue_from_run,
        args.conserve_memory,
        edit_token = args.edit_token,
        edit_location = args.edit_location,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        edit_file=args.edit_file,
        inference_file = args.inference_file,
        inference_type = args.inference_type,
        max_layer = args.max_layer,
        layer_size = args.layer_size,
        v_lr = args.v_lr,
        v_num_grad_steps = args.v_num_grad_steps,
        kl_factor = args.kl_factor,
        max_prob = args.max_prob,
        wandb_active= args.wandb_active,
    )
