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

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union

class MLCDatasetsFormatterBase(ABC):
    """
    Base class for processing datasets for MLC Evaluators.
    """

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        """Path to the dataset."""
        self.dataset_path = dataset_path

    @abstractmethod
    def set_annotation_column_name(self):
        """Extract annotations column name from the data."""

    @abstractmethod
    def set_conversation_column_name(self):
        """Extract the text field from the data."""
    
    @abstractmethod
    def set_model_backbone(self):
        """The model backbone for finetuning."""

    @abstractmethod
    def get_training_examples(self, file_path):
        """Extract training examples from the dataset."""
