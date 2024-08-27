# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import json

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
MARKER_TOKEN_LG2 = "[/INST]"
SIZE_OF_MARKER_TOKEN_LG2 = 7

MARKER_TOKEN_LG3 = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
SIZE_OF_MARKER_TOKEN_LG3 = 55 

NUM_VALIDATION_SAMPLES = 1000

class MLCDataset(Dataset):
    """
    Class to prepare convert the training data to tensors, with appropriate masking and yielding."
    """

    def __init__(self, dataset_config, tokenizer, partition="train"):
        """
        Initializes the dataset from config
        """
        self.annotated_data = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.annotated_data = dict(list(self.annotated_data.items())[NUM_VALIDATION_SAMPLES:]) #samples have been randomized at time of creation
        else:
            self.annotated_data = dict(list(self.annotated_data.items())[:NUM_VALIDATION_SAMPLES])
        self.tokenizer = tokenizer

        if dataset_config.model_backbone == "llamaguard2":
            self.marker_token = MARKER_TOKEN_LG2
            self.size_marker_token = SIZE_OF_MARKER_TOKEN_LG2
        elif dataset_config.model_backbone == "llamaguard3":
            self.marker_token = MARKER_TOKEN_LG3
            self.size_marker_token = SIZE_OF_MARKER_TOKEN_LG3

    def __len__(self):
        """Length of the training set"""
        return len(self.annotated_data)

    def __getitem__(self, index):
        """Yielding data items."""
        full_prompt = list(self.annotated_data.items())[index][1]

        # Prompt length
        index_of_instr_enc = full_prompt.find(self.marker_token)
        prompt = full_prompt[: index_of_instr_enc + self.size_marker_token]  # Size of /INST

        output = full_prompt[index_of_instr_enc + self.size_marker_token : -1]
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
