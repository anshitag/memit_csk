#!/usr/bin/env python
# coding: utf-8

# # Script to plot average causal effects
# 
# This script loads sets of hundreds of causal traces that have been computed by the
# `experiment.causal_trace` program, and then aggregates the results to compute
# Average Indirect Effects and Average Total Effects as well as some other information.
# 


import numpy, os
from matplotlib import pyplot as plt
import re

import math
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='gpt2-medium')
parser.add_argument('-e', '--experiment',  default='subject', choices = ['subject', 'verb', 'object'], help='part of speech you want to calculate average for')
parser.add_argument('-f', '--fact_file', default=None, help='path of the prepared data for causal tracing')
parser.add_argument('-i', '--input_dir', type=str, default=None, help='output dir of the causal tracing experiments')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--threshold', type=float, default=None)
args = parser.parse_args()

with open(args.fact_file, 'r') as f:
    fact_json = json.load(f)


# plt.rcParams["font.family"] = "Times New Roman"   
plt.rcParams["mathtext.fontset"] = "dejavuserif"

arch = f"ns3_r0_{args.model}"
archname = args.model.upper()

output_dir = f'{args.input_dir}'
flip_str = '-flip' if args.flip else ''
threshold_str = f'-threshold={args.threshold}' if args.threshold else ''
output_pdf_dir = f'{output_dir}/summary-pdfs{flip_str}{threshold_str}'

COUNT = len(fact_json)

MODEL_LAYERS_MAPPING = {
    'gpt2-medium' : 24,
    'gpt2-large': 36,
    'gpt2-xl': 48
}

NO_LAYERS = MODEL_LAYERS_MAPPING[args.model]

LABELS = [
    "First subject token",
    "Last subject token",
    "Subject",
    "Negative Token",
    "First Verb Token",
    "Last Verb Token",
    "Verb",
    "First Object Token",
    "Last Object Token",
    "Object",
    "Last token",
]

def find_token_range(toks, substring):
    whole_string = "".join(toks)
    char_loc = re.search(rf"\b{substring}\b", whole_string)
    if not char_loc:
        char_loc = whole_string.index(substring)
    else:
        char_loc = char_loc.start()
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        if not self.d:
            return numpy.array([0.0] * NO_LAYERS)
        return numpy.concatenate(self.d).mean(axis=0)

    def std(self):
        if not self.d:
            return numpy.array([0.0] * NO_LAYERS)
        return numpy.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)
    


def filter_func(data, svo_data, args):
    ans = True
    if args.flip:
        ans = ans and (data['answer'] != data['predicted_answer'])
    if args.threshold:
        ans = ans and (data['high_score'] - data['low_score'] > args.threshold)
    return ans


def read_knowlege(count=150, kind=None, arch="gpt2-xl"):
    dirname = f"{output_dir}/cases"
    kindcode = "" if not kind else f"_{kind}"
    (
        avg_first_subject_token,
        avg_last_subject_token,
        avg_subject,
        avg_first_verb_token,
        avg_last_verb_token,
        avg_verb,
        avg_first_obj_token,
        avg_last_obj_token,
        avg_obj,
        avg_neg,
        avg_last_token,
        avg_high_score,
        avg_low_score,
        avg_highest_fixed_score,
    ) = [Avg() for _ in range(14)]

    for i in range(count):
        try:
            data = numpy.load(f"{dirname}/knowledge_{i}{kindcode}.npz")
        except:
            continue
        global filter_func

        # Only consider cases where the model begins with the correct prediction

        svo_data = fact_json[i]
    
        if ("correct_prediction" in data and not data["correct_prediction"]) or not filter_func(data, svo_data, args):
            continue
        input_toks = data["input_tokens"]
        scores = data["scores"]

        first_sub, last_sub = None, None
        if svo_data["subject"]:
            first_sub, last_sub = find_token_range(input_toks, svo_data["subject"])

        first_verb, last_verb = None, None
        if svo_data["verb"]:
            first_verb, last_verb = find_token_range(input_toks, svo_data["verb"])

        first_obj, last_obj = None, None
        if svo_data["object"]:
            first_obj, last_obj = find_token_range(input_toks, svo_data["object"])

        first_neg, last_neg = None, None
        if svo_data["neg"]:
            first_neg, last_neg = find_token_range(input_toks, svo_data["neg"])
        
        last_sub = last_sub - 1
        last_verb = last_verb - 1
        last_obj = last_obj - 1

        last_a = len(scores) - 1

        # original prediction
        avg_high_score.add(data["high_score"])

        # prediction after subject is corrupted
        avg_low_score.add(data["low_score"])
        avg_highest_fixed_score.add(scores.max())

        # First subject, last subjet.
        if first_sub is not None and last_sub is not None:
            avg_first_subject_token.add(scores[first_sub])
            avg_last_subject_token.add(scores[last_sub])
            avg_subject.add_all(scores[first_sub : last_sub+1])


        # Add verb scores
        if first_verb is not None and last_verb is not None:
            avg_first_verb_token.add(scores[first_verb])
            avg_last_verb_token.add(scores[last_verb])
            avg_verb.add_all(scores[first_verb : last_verb+1])

        # Add object scores
        if first_obj is not None and last_obj is not None:
            avg_first_obj_token.add(scores[first_obj])
            avg_last_obj_token.add(scores[last_obj])
            avg_obj.add_all(scores[first_obj : last_obj+1])

        # Add negative scores
        if first_neg is not None and last_neg is not None:
            avg_neg.add_all(scores[first_neg : last_neg])
        avg_last_token.add(scores[last_a])

    result = numpy.stack(
        [
            avg_first_subject_token.avg(),
            avg_last_subject_token.avg(),
            avg_subject.avg(),
            avg_neg.avg(),
            avg_first_verb_token.avg(),
            avg_last_verb_token.avg(),
            avg_verb.avg(),
            avg_first_obj_token.avg(),
            avg_last_obj_token.avg(),
            avg_obj.avg(),
            avg_last_token.avg(),
        ]
    )
    result_std = numpy.stack(
        [
            avg_first_subject_token.std(),
            avg_last_subject_token.std(),
            avg_subject.std(),
            avg_neg.std(),
            avg_first_verb_token.std(),
            avg_last_verb_token.std(),
            avg_verb.std(),
            avg_first_obj_token.std(),
            avg_last_obj_token.std(),
            avg_obj.std(),
            avg_last_token.std(),
        ]
    )
    print(f"Average Total Effect {avg_high_score.avg() - avg_low_score.avg():.3f}")
    print(
        f"Best average indirect effect on first subject {avg_first_subject_token.avg().max() - avg_low_score.avg():.3f}"
    )
    print(
        f"Best average indirect effect on last subject {avg_last_subject_token.avg().max() - avg_low_score.avg():.3f}"
    )
    print(
        f"Best average indirect effect on all subject {avg_subject.avg().max() - avg_low_score.avg():.3f}"
    )

    print(
        f"Best average indirect effect on first verb {avg_first_verb_token.avg().max() - avg_low_score.avg():.3f}"
    )
    print(
        f"Best average indirect effect on last verb {avg_last_verb_token.avg().max() - avg_low_score.avg():.3f}"
    )
    print(
        f"Best average indirect effect on all verb {avg_verb.avg().max() - avg_low_score.avg():.3f}"
    )

    print(
        f"Best average indirect effect on first object {avg_first_obj_token.avg().max() - avg_low_score.avg():.3f}"
    )
    print(
        f"Best average indirect effect on last object {avg_last_obj_token.avg().max() - avg_low_score.avg():.3f}"
    )
    print(
        f"Best average indirect effect on all object {avg_obj.avg().max() - avg_low_score.avg():.3f}"
    )

    print(
        f"Best average indirect effect on negative token {avg_neg.avg().max() - avg_low_score.avg():.3f}"
    )

    print(
        f"Best average indirect effect on last token {avg_last_token.avg().max() - avg_low_score.avg():.3f}"
    )

    return dict(
        low_score=avg_low_score.avg(), result=result, result_std=result_std, size=avg_high_score.size()
    )


def plot_array(
    differences,
    kind=None,
    savepdf=None,
    title=None,
    low_score=None,
    high_score=None,
    archname="GPT2-XL",
):
    if low_score is None:
        low_score = differences.min()
    if high_score is None:
        high_score = differences.max()
    answer = "AIE"

    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    h = ax.pcolor(
        differences,
        cmap={None: "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        vmin=low_score,
        vmax=high_score,
    )
    if title:
        ax.set_title(title)
    ax.invert_yaxis()
    ax.set_yticks([0.5 + i for i in range(len(differences))])
    ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
    ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
    ax.set_yticklabels(LABELS)
    if kind is None:
        ax.set_xlabel(f"single patched layer within {archname}")
    else:
        ax.set_xlabel(f"center of interval of 10 patched {kind} layers")
    cb = plt.colorbar(h)
    # The following should be cb.ax.set_xlabel(answer), but this is broken in matplotlib 3.5.1.
    if answer:
        cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)

    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
    plt.show()


high_score = None  # Scale all plots according to the y axis of the first plot

for kind in [None, "mlp", "attn"]:
    d = read_knowlege(COUNT, kind, arch)
    count = d["size"]
    what = {
        None: "Indirect Effect of $h_i^{(l)}$",
        "mlp": "Indirect Effect of MLP",
        "attn": "Indirect Effect of Attn",
    }[kind]
    title = f"Avg {what} over {count} prompts"
    result = numpy.clip(d["result"] - d["low_score"], 0, None)
    kindcode = "" if kind is None else f"_{kind}"
    if kind not in ["mlp", "attn"]:
        high_score = result.max()
    plot_array(
        result,
        kind=kind,
        title=title,
        low_score=0.0,
        high_score=high_score,
        archname=archname,
        savepdf=f"{output_pdf_dir}/rollup{kindcode}.png",
    )



color_order = [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 1, 2]
x = None

cmap = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True, dpi=200)
for j, (kind, title) in enumerate(
    [
        (None, "single hidden vector"),
        ("mlp", "run of 10 MLP lookups"),
        ("attn", "run of 10 Attn modules"),
    ]
):
    print(f"Reading {kind}")
    d = read_knowlege(COUNT, kind, arch)
    for i, label in list(enumerate(LABELS)):
        y = d["result"][i] - d["low_score"]
        if x is None:
            x = list(range(len(y)))
        std = d["result_std"][i]
        error = std * 1.96 / math.sqrt(count)
        axes[j].fill_between(
            x, y - error, y + error, alpha=0.3, color=cmap.colors[color_order[i]]
        )
        axes[j].plot(x, y, label=label, color=cmap.colors[color_order[i]])

    axes[j].set_title(f"Average indirect effect of a {title}")
    axes[j].set_ylabel("Average indirect effect on p(o)")
    axes[j].set_xlabel(f"Layer number in {archname}")
    # axes[j].set_ylim(0.1, 0.3)
axes[1].legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{output_pdf_dir}/lineplot-causaltrace.png")
plt.show()

