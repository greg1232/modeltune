import argparse
import os
import subprocess
from datetime import datetime

import yaml
from pydantic import BaseModel

from runners.safety_model_config import SafetyModelConfig
from runners.safety_models_runner import SafetyModelsRunnerConfig, run_safety_models
from schemas.safety_datasets import (
    AnnotationRun,
    PseudoGroundTruthDatasetSchema,
    ResponseDataset,
    SafetyModelAnnotationDataset,
)

SAFETY_MODEL_RUN_DIR = "safety_model_run"
ANNOTATION_RECORDS_CSV = "safety_model_annotation_records.csv"
PSEUDO_GROUND_TRUTHS_OUTPUT_CSV = "pseudo_ground_truth_labels.csv"


class PsuedoGroundTruthResponsesRunnerConfig(BaseModel):
    responses_dataset_path: str
    safety_model_config: SafetyModelConfig

    @classmethod
    def from_yml(cls, filepath: str):
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


def main():
    parser = argparse.ArgumentParser(
        description="Get pseudo ground truth labels for a certain dataset"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    print("Loading config...")
    config = PsuedoGroundTruthResponsesRunnerConfig.from_yml(args.config)
    pseudo_gt_label_responses(config)


def pseudo_gt_label_responses(config: PsuedoGroundTruthResponsesRunnerConfig):
    # Create run dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = f"./run_{timestamp}"
    os.makedirs(run_output_dir)

    # Get SUT responses
    response_dataset = ResponseDataset.from_csv(config.responses_dataset_path)

    # Run safety model runner submodule
    safety_models_run_config = SafetyModelsRunnerConfig(
        responses_dataset_path=config.responses_dataset_path,
        safety_model_config=config.safety_model_config,
    )
    safety_model_outputs = run_safety_models(
        safety_models_run_config,
        run_output_dir=os.path.join(run_output_dir, SAFETY_MODEL_RUN_DIR),
    )

    # Join the annotation results into pseudo ground truths
    print(f"Collected all safety model responses.")
    print(f"Combining safety model responses into a record format...")
    annotation_runs = []
    for annotation_filepath in safety_model_outputs.values():
        run = AnnotationRun.from_jsonl(annotation_filepath)
        safety_model_annotation = run.to_safety_model_annotation()
        annotation_runs.append(safety_model_annotation)

    # Create pseudo annotation joined table
    concat_annotation_records = SafetyModelAnnotationDataset.concat(annotation_runs)
    concat_annotation_records.to_csv(
        os.path.join(run_output_dir, ANNOTATION_RECORDS_CSV)
    )
    print(f"Saved combined annotation records to {ANNOTATION_RECORDS_CSV}.")

    print(f"Computing pseudo ground truths...")
    pseudo_gt_dataset = concat_annotation_records.compute_pseudo_ground_truths()
    pseudo_gt_schema = PseudoGroundTruthDatasetSchema()

    print(
        f"Merging pseudo ground truths with response dataset for better data visibility."
    )
    result = response_dataset.df.merge(
        pseudo_gt_dataset.df, on=pseudo_gt_schema.pair_uid, how="left"
    )

    result.to_csv(
        os.path.join(run_output_dir, PSEUDO_GROUND_TRUTHS_OUTPUT_CSV), index=False
    )
    print(
        f"Completed pseudo ground truth computation. Saved to {PSEUDO_GROUND_TRUTHS_OUTPUT_CSV}."
    )


if __name__ == "__main__":
    main()
