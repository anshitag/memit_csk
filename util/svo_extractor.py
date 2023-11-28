import spacy
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--output', type=str, default=None)

args = parser.parse_args()
input = args.input
output = args.output

nlp = spacy.load('en_core_web_lg')

with open(input, 'r') as f:
    data = json.load(f)

sentences = [d["prompt"] for d in data]


out_dir = os.path.dirname(output)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def get_subject_phrase(doc):
    for token in doc:
        if (token.dep_ in ['nsubj', 'compound']):
            # subtree = list(token.subtree)
            # start = subtree[0].i
            # end = subtree[-1].i + 1
            # return ' '.join([tok.text for tok in doc[start:end]])
            return token.text

def get_object_phrase(doc):
    for token in doc:
        if (token.dep_ in ['dobj', 'pobj','iobj', 'attr', 'oprd', 'dative', 'xcomp'] or (token.pos_ in ['NOUN', 'PROPN'] and token.dep_ == 'ROOT')):
            # subtree = list(token.subtree)
            # start = subtree[0].i
            # end = subtree[-1].i + 1
            # return ' '.join([tok.text for tok in doc[start:end]])
            return token.text
        
def get_verb(doc):
    for token in doc:
        if token.pos_ == 'VERB':
            return token.text
    return None

def get_neg(doc):
    for token in doc:
        if token.dep_ == 'neg':
            return token.text
    return None

docs = list(nlp.pipe(sentences))


for i, sent in enumerate(sentences):
    doc = nlp(sent)
    print("Sentence: {}".format(sent))
    sub = get_subject_phrase(doc)
    obj = get_object_phrase(doc)
    verb = get_verb(doc)
    neg = get_neg(doc)
    print(sub, obj, verb)
    if not sub or not obj or not verb:
        arr = sent.split()
        if len(arr) == 3:
            sub = arr[0]
            verb = arr[1]
            obj = arr[2]

    data[i]['subject'] = sub
    data[i]['verb'] = verb
    data[i]['object'] = obj
    data[i]['neg'] = neg

with open(output, 'w+') as f:
    json.dump(data, f, indent = 4)