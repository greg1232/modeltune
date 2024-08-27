# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass
from enum import Enum
class ModelBackbone(Enum):
    """
    Enum for defining the model backbone that may be used for training with this data. 
    """
    LLAMAGUARD2 = "llamaguard2",
    LLAMAGUARD3 = "llamaguard3",
    MISTRAL = "mistralv0_3"

@dataclass
class samsum_dataset:
    """ Original Llama recipes dataset"""
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    """ Original Llama recipes dataset"""
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = (
        "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    )


@dataclass
class alpaca_dataset:
    """ Original Llama recipes dataset"""
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"


@dataclass
class custom_dataset:
    """ Original Llama recipes dataset showcases custom dataset"""
    dataset: str = "custom_dataset"
    file: str = "recipes/quickstart/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class llamaguard_toxicchat_dataset:
    """ Original Llama recipes dataset"""
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class aegis_dataset:
    """Training dataset that uses subset of aegis data"""

    dataset: str = "aegis_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path = "data/llama_guard_training_data/aegis_small_july_0727_aegis_training_data_20240724-231749.json"


@dataclass
class mlc_dataset:
    """Training dataset that uses mlc V0.5 ~40k AI labelled dataset"""

    dataset: str = "mlc_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path = "data/llama_guard_training_data/32k_merged_mlc_v0_5_teacher_annotations_0724_mlc_training_data_20240726-030542.json"
    model_backbone = ModelBackbone.LLAMAGUARD2.value[0]

@dataclass
class mlc_dataset_lg3:
    """Training dataset that uses mlc V0.5 ~40k AI labelled dataset"""

    dataset: str = "mlc_dataset_lg3"
    train_split: str = "train"
    test_split: str = "val"
    data_path = "data/llama_guard_training_data/Preprocessed_32k_mlc_v0_5_teacher_annotations_0723_mlc_training_data_20240820-020851.json"
    model_backbone = ModelBackbone.LLAMAGUARD3.value[0]

