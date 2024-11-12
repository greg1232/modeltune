import argparse
import os
import subprocess
from datetime import datetime
from typing import List

import pandas as pd
import yaml
from pydantic import BaseModel

from measure.measure_safety_model import SafetyModelMeasurementRun, measure
from runners.safety_model_config import SafetyModelConfig
from runners.safety_models_runner import SafetyModelsRunnerConfig, run_safety_models
from schemas.safety_datasets import AnnotationRun, MeasurementGroundTruthDataset

run_output_dir = "./run_output/"
GT_INPUT_PATH = "gt_input.csv"
ANNOTATOR_OUTPUT_DIR = "annotations"
ANNOTATOR_INPUT_CSV = "annotator_input.csv"
METRICS_OUTPUT_DIR = "measurement_results"
SCORES_CSV = "scores.csv"

RESPONSES_CSV = "sut_responses.csv"
SAFETY_MODEL_RUN = "safety_model_run"


class MeasurementRunnerConfig(BaseModel):
    working_dir: str = "./"
    measurement_ground_truth_path: str
    safety_model_config: SafetyModelConfig

    @classmethod
    def from_yml(cls, filepath: str):
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


def main():
    parser = argparse.ArgumentParser(description="Run safety model measurement.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    config = MeasurementRunnerConfig.from_yml(args.config)
    measure_safety_model(config)


def measure_safety_model(config: MeasurementRunnerConfig):
    # Set working dir
    os.chdir(config.working_dir)

    # Create run dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = f"./run_{timestamp}"
    os.makedirs(run_output_dir)

    # Load ground truth file
    gt_dataset = MeasurementGroundTruthDataset.from_csv(
        config.measurement_ground_truth_path
    )

    # Convert to input ready annotation input file
    measurement_gt_input_fpath = os.path.join(run_output_dir, GT_INPUT_PATH)
    gt_dataset.to_csv(measurement_gt_input_fpath)

    # Create responses file to submit to safety model runner
    responses_filepath = os.path.join(run_output_dir, RESPONSES_CSV)
    responses_dataset = gt_dataset.to_responses_dataset()
    responses_dataset.to_csv(responses_filepath)

    # Run safety model runner submodule
    safety_models_run_config = SafetyModelsRunnerConfig(
        responses_dataset_path=responses_filepath,
        safety_model_config=config.safety_model_config,
    )
    safety_model_outputs = run_safety_models(
        safety_models_run_config,
        run_output_dir=os.path.join(run_output_dir, SAFETY_MODEL_RUN),
    )

    # Compute measurements
    measurements: List[SafetyModelMeasurementRun] = []
    for safety_model, safety_model_annotations_filepath in safety_model_outputs.items():
        # TODO actually control the output filename
        m = measure(
            ground_truth_file=measurement_gt_input_fpath,
            annotator_run_data_file=safety_model_annotations_filepath,
            output_folder=os.path.join(run_output_dir, METRICS_OUTPUT_DIR),
        )
        measurements.append(m)

    scores = [{"safety_model": m.safety_model, **m.scores} for m in measurements]
    scores_df = pd.json_normalize(scores)

    scores_output_path = os.path.join(run_output_dir, SCORES_CSV)
    print(f"Writing scores to {scores_output_path}...")
    scores_df.to_csv(scores_output_path, index=False)

    print("Measurement runner completed all tasks.")


if __name__ == "__main__":
    main()
