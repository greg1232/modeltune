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

import argparse
import json
import os


from pathlib import Path
import time


from model.mlc_llama_guard.meta_llama_recipes.src.llama_recipes.data.llama_guard.mlc_data.mlc_data_format import (
    MLC_TAXONOMY_0_5,
    AgentType,
    formatter_configs,
    formatter_configs_lg3,
    ModelBackbone
)

from model.mlc_llama_guard.meta_llama_recipes.src.llama_recipes.data.llama_guard.mlc_finetuning_data_formatter import (
    MLCDatasetsFormatterBase,
)

from model.mlc_llama_guard.meta_llama_recipes.src.llama_recipes.data.llama_guard.finetuning_data_formatter import (
    TrainingExample,
    create_formatted_finetuning_examples,
)


class MLCDataFormatter(MLCDatasetsFormatterBase):
    """Class responsible for MLC V0.5 dataset creation for Llama Guard Finetuning.

    Args:
        MLCDatasetsFormatterBase (Type[MLCDatasetsFormatterBase]): abstract base class for data processing.
    """

    def __init__(self, dataset_path: str | Path) -> None:
        super().__init__(dataset_path)
        self.labels_column_name = None
        self.conversation_column_name = None

    def set_annotation_column_name(self, column_name):
        """Pass the name of the column/field for annotations/labels ."""
        self.labels_column_name = column_name

    def set_conversation_column_name(self, column_name):
        """Pass the name of the column/field for the text/conversation."""
        self.conversation_column_name = column_name

    def set_model_backbone(self, model_backbone):
        """Pass the model backbone name for appropiate formatting"""
        self.model_backbone = model_backbone

    def get_training_examples(self, file_path):
        """Extract training examples with proper labels and category codes."""
        assert file_path, "Invalid filepath"

        base_filename = os.path.basename(file_path)
        base_name, _ = base_filename.split(".json")

        # Output file for serializing Llama Guard training ready data
        write_to_file = {}

        training_examples = []
        cnt_resp = 0
        cnt_prompt = 0
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            # Format expected by Llama Guard for prompt and response
            response = "N/A"
            agent_type = ""
            assert isinstance(json_data, list), "Expecting a list"
            for i, sample in enumerate(json_data):
                assert self.labels_column_name, "Invalid labels column name provided"
                annotations = sample[self.labels_column_name]

                assert "safety_label_training" in sample, "Safety label field missing"
                label = sample["safety_label_training"]

                # Map the taxonomy to category IDs to be compatible with Llama Guard
                cat_codes = [
                    MLC_TAXONOMY_0_5[annot]
                    for annot in annotations
                    if annot in MLC_TAXONOMY_0_5
                ]
                cat_codes = list(set(cat_codes))

                # Extract the conversation from dataset
                assert self.conversation_column_name, "Invalid conversation column name"
                conversation = sample[self.conversation_column_name]

                if len(conversation) > 2:
                    # for multiturn, remove the last response
                    prompt = f'{conversation[0]["content"]}'
                    conversation = conversation[1:]

                    new_turns = []
                    i = 1
                    for i, turn in enumerate(conversation):
                        # user turn when odd
                        if ((i + 1) % 2) == 0:
                            user_msg = f'user: {turn["content"]}'
                            new_turns.append(user_msg)

                        # agent turn
                        elif ((i + 1) % 2) != 0:
                            bot_msg = f'agent: {turn["content"]}'
                            new_turns.append(bot_msg)
                    response = "\n\n".join(new_turns)
                    agent_type = AgentType.AGENT.value
                    cnt_resp += 1

                # Single full turn
                elif len(conversation) == 2:
                    prompt = f'{conversation[0]["content"]}'
                    response = f'agent: {conversation[1]["content"]}'

                    if response != "":
                        agent_type = AgentType.AGENT.value
                    else:
                        agent_type = AgentType.USER.value
                    cnt_resp += 1

                # Prompt only
                elif len(conversation) == 1:
                    # prompt = conversation[0]['content']
                    prompt = f'{conversation[0]["content"]}'
                    response = "N/A"
                    agent_type = AgentType.USER.value
                    cnt_prompt += 1

                tr_example = TrainingExample(
                    prompt=prompt,
                    response=response,
                    agent_type=agent_type,
                    violated_category_codes=cat_codes,
                    label=label,
                    explanation="",
                )

                training_examples.append(tr_example)

                write_to_file[i] = {
                    "prompt": prompt,
                    "response": response,
                    "violated_category_codes": cat_codes,
                    "label": label,
                    "explanation": "",
                }
        return training_examples, write_to_file

    def get_formatted_training_examples(self, training_examples):
        """Method to format the training examples for finetuning"""
        formatted_examples = []

        # Call the create_formatted_finetuning_examples function

        if self.model_backbone == ModelBackbone.LLAMAGUARD2.value[0]:
            formatted_examples = create_formatted_finetuning_examples(
            training_examples, formatter_configs
        )
        elif self.model_backbone == ModelBackbone.LLAMAGUARD3.value[0]:
            formatted_examples = create_formatted_finetuning_examples(
            training_examples, formatter_configs_lg3
        )
        else:
            raise ValueError("Unrecognized backbone")
        
        return formatted_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create training examples for MLC dataset."
    )
    parser.add_argument(
        "--file_path", required=True, type=str, help="Path to MLC dataset."
    )
    parser.add_argument(
        "--label_column",
        required=True,
        type=str,
        help="Column name for labels/annotations.",
    )
    parser.add_argument(
        "--text_column",
        required=True,
        type=str,
        help="Column name for text/conversation.",
    )
    parser.add_argument(
        "--model_backbone",
        required=True,
        type=str,
        help="Model backbone for finetuning.",
    )
    args = parser.parse_args()

    assert args.file_path, "invalid file path"
    mlc_formatter = MLCDataFormatter(args.file_path)
    assert mlc_formatter, "Invalid Dataset formatter object"
    

    mlc_formatter.set_annotation_column_name(args.label_column)
    mlc_formatter.set_conversation_column_name(args.text_column)
    mlc_formatter.set_model_backbone(args.model_backbone)

    training_examples, training_examples_serialized = (
        mlc_formatter.get_training_examples(args.file_path)
    )

    assert len(training_examples) != 0, "No training examples to format"
    # Call the create_formatted_finetuning_examples function
    formatted_examples = create_formatted_finetuning_examples(
        training_examples, formatter_configs
    )
    formatted_examples = mlc_formatter.get_formatted_training_examples(training_examples)

    base_filename = os.path.basename(args.file_path)
    base_name, ext = base_filename.split(".json")

    # File to write Llama Guard training ready data
    output_dir_name = "./data/llama_guard_training_data"
    os.makedirs(output_dir_name, exist_ok=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = os.path.join(
        output_dir_name,
        f"{base_name}_mlc_training_data_{timestr}.json",
    )
    formatted_ex_file = open(output_file_name, "w")

    format_data_towrite = {}
    for i, formatted_eg in enumerate(formatted_examples):
        format_data_towrite[i] = formatted_eg

    json.dump(format_data_towrite, formatted_ex_file, ensure_ascii=False, indent=4)
    formatted_ex_file.close()
