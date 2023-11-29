import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=None)

args = parser.parse_args()

with open(args.input, 'r') as f:
    data = json.load(f)
for d in data:
    if d['subject'] == d['object'] or d['object'] == d['verb'] or d['subject'] == d['verb']:
        print(f"Invalid SVO for instance: \n{d}")