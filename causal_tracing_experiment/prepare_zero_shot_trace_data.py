import argparse
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os
import json
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='gpt2-xl')
parser.add_argument('-i', '--input', default=None)
parser.add_argument('-svo', '--svo', default=None)
parser.add_argument('-o', '--output', default=None)
parser.add_argument('-p', '--positive', default='True')
parser.add_argument('-n', '--negative', default='False')
parser.add_argument('-b', '--batch_size', type=int, default=1)


args = parser.parse_args()

MODEL = args.model
INPUT_FILE = args.input
SVO_FILE = args.svo
OUTPUT_FILE = args.output

BATCH_SIZE = args.batch_size

def chunks(arr1, arr2, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr1), n):
        yield arr1[i : i + n], arr2[i : i + n]


pos_token, neg_token = args.positive, args.negative

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpt2 = AutoModelForCausalLM.from_pretrained(MODEL, return_dict_in_generate=True).to(device)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL, padding_side = "left")
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token


with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

with open(SVO_FILE, 'r') as f:
    svo_data = json.load(f)

question_prompt = f'{pos_token} or {neg_token}?'
all_prompts = [text["prompt"] + '. ' +  question_prompt for text in data]
labels = [m["label"] for m in data]
del data


p_index, n_index = tokenizer(f' {pos_token}')["input_ids"][0], tokenizer(f' {neg_token}')["input_ids"][0]
print(all_prompts[:3])
print('positive token', pos_token, 'index', p_index)
print('negative token', neg_token, 'index', n_index)

pred_labels = []
pred_probs = []
for prompts, labs in chunks(all_prompts, labels, BATCH_SIZE):
    encoded_input = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = gpt2(**encoded_input).logits
        logits = logits[:, -1, :]
        probs = logits.softmax(dim = 1)
        del logits

        target_indexes = torch.tensor([n_index, p_index]).view(1, -1).repeat_interleave(repeats=probs.size(0), dim=0).to(device)
        # print(target_indexes)
        false_true_probs = torch.gather(
            probs,
            1,
            target_indexes
        )
        preds = false_true_probs.argmax(dim=-1).cpu().numpy().tolist()
        pred_labels += preds
        pred_probs += false_true_probs[:, 1].cpu().numpy().tolist()
        torch.cuda.empty_cache()

count = 0
out = []

LABEL_MAP = { 0: neg_token, 1: pos_token }

for i in range(len(all_prompts)):
    prompt = all_prompts[i]
    label = labels[i]
    pred_label = pred_labels[i]

    if label != pred_label:
        continue
    svo = svo_data[i]
    if svo["subject"] and svo["verb"] and svo["object"]:
        out.append({
            'prompt': prompt,
            'attribute': LABEL_MAP[pred_label],
            "known_id": count,
            "subject": svo["subject"],
            "verb": svo["verb"],
            "object": svo["object"],
            "neg": svo["neg"],
        })
        count +=1 
    
# print('Accuracy = {}%'.format(count / len(all_prompts) * 100))
print('Accuracy = ', metrics.accuracy_score(labels, pred_labels))

conf_matrix  = confusion_matrix(labels, pred_labels, labels=[True, False])
print('Confusion Matrix = ')
print(conf_matrix)
print('F1 score = ', metrics.f1_score(labels, pred_labels, average='weighted', labels=[1, 0]))
fpr, tpr, thresholds = metrics.roc_curve(labels, pred_probs, pos_label=1)
print('AUC score = ', metrics.auc(fpr, tpr))

print('Classification Report: \n',  metrics.classification_report(labels, pred_labels, labels=[0, 1], zero_division=1))
with open(OUTPUT_FILE, 'w') as f:
    json.dump(out, f, indent=4)