import argparse, os, json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from sklearn import metrics
import numpy as np

class CommonSenseDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        tok: AutoTokenizer,
        eval_dataset_type: str,
        dict_label: dict,
        max_length: int = 32
    ):
        data_dir = Path(data_dir)
        with open(data_dir, "r") as f:
            self.data = json.load(f)

        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for e in self.data:
            if eval_dataset_type == 'efficacy':
                prompt = e["prompt"] + ":"
                tok_output = tok(prompt, max_length=max_length, padding="max_length", truncation=False)
                self.input_ids.append(torch.tensor(tok_output['input_ids']))
                self.attention_mask.append(torch.tensor(tok_output['attention_mask']))
                self.labels.append(e["label"])
            else:
                for i in e[eval_dataset_type]:
                    prompt = i + ":"
                    tok_output = tok(prompt, max_length=max_length, padding="max_length", truncation=False)
                    self.input_ids.append(torch.tensor(tok_output['input_ids']))
                    self.attention_mask.append(torch.tensor(tok_output['attention_mask']))
                    if eval_dataset_type == "affected_reasoning":
                        self.labels.append(dict_label[1])
                    else:
                        self.labels.append(e["label"])

        print(f"Loaded {eval_dataset_type} dataset with {len(self)} elements")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_mask[item]

    def getlabel(self):
        return self.labels


def next_word_prediction(model, dataloader, tok):

    pred_label = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            colon_indexes = batch['attention_mask'].sum(dim = -1) - 1
            colon_indexes = colon_indexes.view(-1, 1)
            out = model(**batch).logits.argmax(dim=-1)
            prediction = torch.gather(out, 1, colon_indexes)
            pred_label += tok.batch_decode(prediction)
    return pred_label


def calculate_metrics(orig_label, pred_label, eval_type, labels):

    acc = metrics.accuracy_score(orig_label, pred_label)
    f1 = metrics.f1_score(orig_label, pred_label, average='weighted', labels=labels, zero_division=1)
    print(f'{eval_type} Accuracy = ', acc)
    print(f'{eval_type} F1 score = ', f1)
    try:
        print(f'{eval_type} Confusion Matrix = \n', metrics.confusion_matrix(orig_label, pred_label, labels=labels))
    except:
        print('Confusion matrix cannot be calculated')
    print('{eval_type} Classification Report: \n',  metrics.classification_report(orig_label, pred_label, labels=labels, zero_division=1))
    return {"{eval_type} accuracy": acc, "{eval_type} f1Score": f1}


def evaluate(model, MODEL, DATA_FILE, BATCH_SIZE, dict_label, device):

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="right")
    tok.pad_token = tok.eos_token

    model.eval()

    eval_types = ['efficacy', 'affected_reasoning', 'affected_neighborhood_subject', 'affected_neighborhood_object', 'affected_neighborhood_verb', 'affected_paraphrase']

    for e_type in eval_types:
        dataset = CommonSenseDataset(DATA_FILE, tok, e_type, dict_label)

        data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data])}

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

        p_index, n_index, other_index = dict_label[1], dict_label[0], "None"

        pred_label = next_word_prediction(model, dataloader, tok)
        true_pred = dataset.getlabel()
        prediction_label = [other_index if i not in dict_label.values() else i for i in pred_label]
        split_metrics = calculate_metrics(true_pred, prediction_label, e_type, [p_index, n_index, other_index])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='gpt2-large')
    parser.add_argument('-mp', '--model_path', type=str)
    parser.add_argument('-i', '--input_file', type=str)
    parser.add_argument('-b', '--log2_batch_size', type=int, default=6)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dict_label = {1: " True", 0: " False"}
    MODEL = args.model
    BATCH_SIZE = pow(2, args.log2_batch_size)
    MODEL_PATH = args.model_path
    DATA_FILE = args.input_file

    model = None
    try: 
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    except:
        print(f"No saved model found at this path: {MODEL_PATH}")
        exit()

    evaluate(model, MODEL, DATA_FILE, BATCH_SIZE, dict_label, device)