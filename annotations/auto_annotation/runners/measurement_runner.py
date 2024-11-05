import argparse
import os
import subprocess
from typing import List

import pandas as pd
import yaml
from pydantic import BaseModel

from measure.measure_safety_model import SafetyModelMeasurementRun, measure
from runners.safety_model_config import SafetyModelConfig
from schemas.safety_datasets import AnnotationRun, MeasurementGroundTruthDataset

RUN_OUTPUT_DIR = "./run_output/"
GT_INPUT_PATH = "gt_input.csv"
ANNOTATOR_OUTPUT_DIR = "annotations"
ANNOTATOR_INPUT_CSV = "annotator_input.csv"
METRICS_OUTPUT_DIR = "measurement_results"
SCORES_CSV = "scores.csv"


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
    if os.path.exists(RUN_OUTPUT_DIR):
        raise FileExistsError(f"The directory '{RUN_OUTPUT_DIR}' already exists.")
    os.makedirs(RUN_OUTPUT_DIR)

    # Load ground truth file
    gt_dataset = MeasurementGroundTruthDataset.from_csv(
        config.measurement_ground_truth_path
    )

    # Convert to input ready annotation input file
    measurement_gt_input_fpath = os.path.join(RUN_OUTPUT_DIR, GT_INPUT_PATH)
    gt_dataset.to_csv(measurement_gt_input_fpath)

    # Generate annotation input files
    annotator_output_dir = os.path.join(RUN_OUTPUT_DIR, ANNOTATOR_OUTPUT_DIR)
    if os.path.exists(annotator_output_dir):
        raise FileExistsError(f"The directory '{annotator_output_dir}' already exists.")
    os.makedirs(annotator_output_dir)
    annotator_input_filepath = os.path.join(annotator_output_dir, ANNOTATOR_INPUT_CSV)
    gt_dataset.write_annotation_input_csv(annotator_input_filepath)

    # Run annotations
    for a in config.safety_model_config.safety_models:
        # check for a run file, if it already exists, then load that and skip the annotation
        command = [
            "poetry",
            "run",
            "modelgauge",
            "run-csv-items",
            "-a",
            a,
            annotator_input_filepath,
        ]
        result = subprocess.run(command, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"Error running modelgauge for model {a}: {result.stderr}")
            exit(1)
        else:
            print(f"Successfully ran modelgauge for model {a}: {result.stdout}")

    annotation_fnames = []
    for filename in os.listdir(annotator_output_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(annotator_output_dir, filename)
            run = AnnotationRun.from_jsonl(filepath)

            # Check that all rows were generated (no silent failures... no way to tell other than double checking)
            if len(run) != len(gt_dataset):
                raise ValueError(
                    f"Annotation count for {filename} does not match input dataset count. Annotation count: {len(run)}. Ground truth dataset count: {len(gt_dataset)}"
                )

            annotation_fnames.append(filepath)

    measurements: List[SafetyModelMeasurementRun] = []
    for annotation_f in annotation_fnames:
        m = measure(
            ground_truth_file=measurement_gt_input_fpath,
            annotator_run_data_file=annotation_f,
            output_folder=os.path.join(RUN_OUTPUT_DIR, METRICS_OUTPUT_DIR),
        )
        measurements.append(m)

    scores = [{"safety_model": m.safety_model, **m.scores} for m in measurements]
    scores_df = pd.json_normalize(scores)

    scores_output_path = os.path.join(RUN_OUTPUT_DIR, SCORES_CSV)
    print(f"Writing scores to {scores_output_path}...")
    scores_df.to_csv(scores_output_path, index=False)

    print("Measurement runner completed all tasks.")


if __name__ == "__main__":
    main()
