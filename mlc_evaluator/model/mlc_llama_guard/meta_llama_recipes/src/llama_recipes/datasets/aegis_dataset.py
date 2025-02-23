# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Authors: Shaona Ghosh

import copy
import json

import torch
from torch.utils.data import Dataset

SIZE_OF_INST_TOKEN = 7
IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


class AegisDataset(Dataset):
    """
    Class to prepare convert the training data to tensors, with appropriate masking and yield."
    """

    def __init__(self, dataset_config, tokenizer, partition="train"):
        """
        Initializes the dataset from config
        """
        self.annotated_data = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.annotated_data = dict(list(self.annotated_data.items())[400:])
        else:
            self.annotated_data = dict(list(self.annotated_data.items())[:400])
        self.tokenizer = tokenizer

    def __len__(self):
        """Length of the training set"""
        return len(self.annotated_data)

    def __getitem__(self, index):
        full_prompt = list(self.annotated_data.items())[index][1]

        # Prompt length
        index_of_instr_enc = full_prompt.find("[/INST]")
        prompt = full_prompt[: index_of_instr_enc + SIZE_OF_INST_TOKEN]  # Size of /INST

        output = full_prompt[index_of_instr_enc + SIZE_OF_INST_TOKEN : -1]
        example = prompt + output

        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)

        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }
