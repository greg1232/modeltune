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

"""
Takes as input the artifact file generated after running evaluations on golden evaluation sets for V0.5, V1.0 and and visualizes the correlation between evaluator prediction and 
the human ground truth label
"""

import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from argparse import ArgumentParser
from enum import Enum

class ModelBackbone(Enum):
    """
    Enum for defining the model that may be members of the ensemble
    """
    LLAMAGUARD2 = "llamaguard2",
    LLAMAGUARD3 = "llamaguard3",
    LLAMA_70B = "llama_3_70b"    
    MIXTRAL_22B = "mistral_8x22b"
    CUSTOM = "vllm" #TODO this is a placeholder in production pipeline for now


def write_resuts_tofile(base_name, expt_name, data):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_file_name = os.path.join(
        "./analysis/",
        f"{base_name}_{expt_name}_{timestr}.json",
    )
    formatted_ex_file = open(output_file_name, "w")

    format_data_towrite = {}
    for i, formatted_eg in enumerate(data):
        format_data_towrite[i] = formatted_eg

    json.dump(format_data_towrite, formatted_ex_file, ensure_ascii=False, indent=4)
    formatted_ex_file.close()

def analyze_results(file_path, output_filename, taxonomy_subset):
    if os.path.isfile(file_path):
        assert file_path, "Invalid results artifacts file."

        base_filename = os.path.basename(file_path)
        basename, _ = base_filename.split(".json")

        # Output file for serializing Llama Guard training ready data
        write_to_file = {}
        try:
            with open(file_path , 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                len_samples = len(json_data)
                assert len_samples != 0, "Results artifact file empty" 

                ensemble_preds = []
                ensemble_invalid_cnt = 0
                gt_labels = []

                all_experts_wrong_unsafe_cnt = 0
                all_experts_wrong_safe_cnt = 0
                all_experts_wrong_unsafe_vector = [1,1,1,1,0]
                all_experts_wrong_safe_vector = [0,0,0,0,1]
                samples_all_experts_unsafe = []
                samples_all_experts_safe = []

                labels_per_model = dict()
                
                # Validate if the expected model predictions are present
                assert "safety_model_response" in json_data["tests"][0], "Ensemble prediction not available in results artifact file"
                assert "ground_truth" in json_data["tests"][0], "No ground truth available in results artifact file"

                model_responses = json_data["tests"][0]["safety_model_response"]["joined_responses"]
                assert ModelBackbone.MIXTRAL_22B.value in model_responses, \
                                        f"Missing evaluator {ModelBackbone.MIXTRAL_22B.value} results in results artifact files"
                assert ModelBackbone.LLAMA_70B.value in model_responses, \
                                        f"Missing evaluator {ModelBackbone.LLAMA_70B.value} results in results artifact files"
                assert ModelBackbone.CUSTOM.value in model_responses, \
                                        f"Missing evaluator {ModelBackbone.CUSTOM.value} results in results artifact files"
                for model in model_responses:
                    labels_per_model[model] = list()

                for index, test in enumerate(json_data["tests"]):
                    sample_vector = []
                    uid = test["uid"]
                    if taxonomy_subset == 'v0.5' and not uid.startswith("mlc_ais"):
                        # Only consider samples for V0.5
                        continue
                    elif taxonomy_subset == 'v1.0' and not uid.startswith("id_"):
                        # Only consider samples for V1.0
                        continue
                    
                    ensemble_response = test["safety_model_response"]["is_safe"]
                    ensemble_valid =  test["safety_model_response"]["is_valid"]
                    
                    # Add the ensemble label for analysis
                    if ensemble_valid != True:
                        ensemble_preds.append(None)     
                    else:
                        if ensemble_response == True:
                            ensemble_preds.append(0)
                            sample_vector.append(0)
                        else:
                            ensemble_preds.append(1)
                            sample_vector.append(1)
                    

                    joined_responses = test["safety_model_response"]["joined_responses"]                       
                    for model in joined_responses:
                        if joined_responses[model]["is_valid"] != True:
                            labels_per_model[model].append(None)
                        else:
                            if joined_responses[model]["is_safe"] == True:
                                labels_per_model[model].append(0)
                                sample_vector.append(0)
                            else:
                                labels_per_model[model].append(1)
                                sample_vector.append(1)
                    
                    # Ground truth extraction 
                    gt = test["ground_truth"]["joined_ground_truth"]
                    if gt["is_safe"] == True:
                        gt_labels.append(0)
                        sample_vector.append(0)
                    else:
                        gt_labels.append(1)
                        sample_vector.append(1)


                    # For logging to file
                    sample_wrong_dict = {}
                    sample_wrong_dict["uid"] = test["uid"]
                    sample_wrong_dict["prompt"] = test["prompt"]
                    sample_wrong_dict["response"] = test["response"]
                    if test["uid"].startswith("mlc"):
                        sample_wrong_dict["taxonomy"] = "v0.5"
                    else:
                        sample_wrong_dict["taxonomy"] = "v1.0"
                    sample_wrong_dict["ensemble_pred_is_safe"] = test["safety_model_response"]["is_safe"]
                    sample_wrong_dict["llama3_70b_pred_is_safe"] = test["safety_model_response"]["joined_responses"]["llama_3_70b"]["is_safe"]
                    sample_wrong_dict["llama3_70b_pred_is_safe_reason"] = test["safety_model_response"]["joined_responses"]["llama_3_70b"]["reasoning"]
                    sample_wrong_dict["mistral_8x22b_pred_is_safe"] = test["safety_model_response"]["joined_responses"]["mistral_8x22b"]["is_safe"]
                    sample_wrong_dict["mistral_8x22b_pred_is_safe_reason"] = test["safety_model_response"]["joined_responses"]["mistral_8x22b"]["reasoning"]
                    sample_wrong_dict["mlc_v05_lg2_ft_pred_is_safe"] = test["safety_model_response"]["joined_responses"]["vllm"]["is_safe"]
                    
                    sample_wrong_dict["ground_truth"] = test["ground_truth"]

                    # Check where all experts are wrong at the same time
                    if sample_vector == all_experts_wrong_unsafe_vector:
                        
                        all_experts_wrong_unsafe_cnt += 1
                        samples_all_experts_unsafe.append(sample_wrong_dict)

                    elif sample_vector == all_experts_wrong_safe_vector:
                        all_experts_wrong_safe_cnt += 1 
                        samples_all_experts_safe.append(sample_wrong_dict)

                assert len(gt_labels) == len(ensemble_preds), "Invalid number of evaluator predictions and gt labels"

                plot_ready_predictions = list()
                x_axis_obs = list()
                for model in joined_responses:
                    plot_ready_predictions.append(labels_per_model[model])
                    if model.startswith("mistral"):
                        x_axis_obs.append("mixtral-8x22b")
                    elif model.startswith("vllm"):
                        x_axis_obs.append("mlcv0.5_lg2_ft")
                    else:
                        x_axis_obs.append(model)
                plot_ready_predictions.append(ensemble_preds)
                x_axis_obs.append("Ensemble")
                plot_ready_predictions.append(gt_labels)
                x_axis_obs.append("GroundTruth")

                 # Get samples which are not none for each list in labels_to_plot
                plot_ready_predictions_arr = np.array(plot_ready_predictions)
                samples_with_no_pred = np.equal(plot_ready_predictions_arr, None).any(axis=0)
                plot_ready_predictions_arr = plot_ready_predictions_arr[:, ~samples_with_no_pred].astype(float)

                print(f"All Experts wrong unsafe percentage: {all_experts_wrong_unsafe_cnt/plot_ready_predictions_arr.shape[1]}")
                print(f"All Experts wrong safe percentage: {all_experts_wrong_safe_cnt/plot_ready_predictions_arr.shape[1]}")

                correlation_matrix = np.corrcoef(plot_ready_predictions_arr)
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                plt.title(f"Correlation between Evaluators and Human Ground Truth: {taxonomy_subset}")
                ticks = [x + 0.5 for x in range(len(x_axis_obs))]
                plt.xticks(ticks=ticks, labels=x_axis_obs, rotation=45)
                plt.yticks(ticks=ticks, labels=x_axis_obs, rotation=45)
                assert output_filename, "Invalid output filename provided"
                plt.savefig(output_filename, bbox_inches='tight')

                # Save analysis to files
                write_resuts_tofile(basename, "all_unsafe", samples_all_experts_unsafe)
                write_resuts_tofile(basename, "all_safe", samples_all_experts_safe)

        except Exception as exc:
            raise ValueError("Possible error in artifacts file generated after evaluation runs.") from exc


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--eval_artifact_file", type=str, required=True, help="Artifact json for golden evaluation run.")
    arg_parser.add_argument("--output_filename", type=str, required=True, help="Output filename for results.")
    arg_parser.add_argument("--taxonomy-subset", type=str, required=True, help="Which subset based on taxonomy to run the analysis on.", choices=['v0.5', 'v1.0', 'all'] )
    
    args = arg_parser.parse_args()
    analyze_results(args.eval_artifact_file, args.output_filename, args.taxonomy_subset)