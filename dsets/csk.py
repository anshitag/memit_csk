import json
import typing
from pathlib import Path

import re
from torch.utils.data import Dataset

import random

random.seed(0)

class CommonSenseDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        noise_token: str,
        k_sample_size: int = None,
        *args,
        **kwargs,
    ):
        cf_loc = Path(data_dir)
        self.indexes = None

        with open(cf_loc, "r") as f:
            self.data = json.load(f)
        if k_sample_size is not None:
            print(f"Creating sample dataset of size: {k_sample_size}")
            self.indexes = random.sample(range(len(self.data)), k_sample_size)
            samples = []
            for idx in self.indexes:
                samples.append(self.data[idx])
            self.data = samples

        for i in self.data:
            substring = i["requested_rewrite"][noise_token]
            char_loc = re.search(rf"\b{substring}\b", i["requested_rewrite"]["prompt"])
            if not char_loc:
                i["requested_rewrite"]["prompt"] = i["requested_rewrite"]["prompt"].replace(substring, "{}")
            else:
                i["requested_rewrite"]["prompt"] = re.sub(rf"\b{substring}\b", "{}", i["requested_rewrite"]["prompt"])
            
        print(f"Loaded dataset with {len(self)} elements {self.data[0]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def getindexes(self):
        return self.indexes