import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='path to the output json file after model inference')
parser.add_argument('-svo', '--svo', type=str, help='path to the file containing SVO information')
parser.add_argument('-o', '--output', type=str, help='path to the output file')
args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)

with open(args.svo, 'r') as f:
    svo_data = json.load(f)

outdir = os.path.dirname(args.output)
if outdir and not os.path.exists(outdir):
    os.makedirs(outdir)

out = []
count = 0
for i, d in enumerate(data):
    prompt = d['prompt']
    label = d['label'].strip()
    pred_label = d['predicted_label'].strip()
    # print(prompt, label, pred_label)
    if label != pred_label:
        continue
    svo = svo_data[i]
    if svo["subject"] and svo["verb"] and svo["object"]:
        out.append({
            'prompt': prompt,
            'attribute': pred_label,
            "known_id": count,
            "subject": svo["subject"],
            "verb": svo["verb"],
            "object": svo["object"],
            "neg": svo["neg"],
        })
        count +=1 

with open(args.output, 'w') as f:
    json.dump(out, f, indent=4)