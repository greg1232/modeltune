import os
import subprocess
from typing import List

import yaml
from pydantic import BaseModel

from runners.safety_model_config import SafetyModelConfig
from schemas.safety_datasets import (
    AnnotationRun,
    PseudoGroundTruthDatasetSchema,
    ResponseDataset,
    SafetyModelAnnotationDataset,
)

OUTPUT_DIR = "./run_output"
ANNOTATOR_OUTPUT_DIR = "annotations"
CONFIG_PATH = "./sample_resources/sample_pseudo_gt_config.yml"


class PsuedoGroundTruthResponsesRunnerConfig(BaseModel):
    responses_dataset_path: str
    safety_model_config: SafetyModelConfig

    @classmethod
    def from_yml(cls, filepath: str):
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


def main():
    config = PsuedoGroundTruthResponsesRunnerConfig.from_yml(CONFIG_PATH)
    pseudo_gt_label_responses(config)


def pseudo_gt_label_responses(config: PsuedoGroundTruthResponsesRunnerConfig):
    # TODO validate environment is properly setup
    # Get SUT responses
    response_dataset = ResponseDataset.from_csv(config.responses_dataset_path)

    # TODO based on the annotator_config, verify that the right ENV vars are present

    # Transform the response dataset into the intermediate formate needed for
    # the annotators
    # TODO control the output file location of run-csv-items. We need to change
    # modelgauge code
    # Workaround: We know the outputs are in the same dir as the input file, so
    # putting the input file in a subdir, we can control the output location
    # slightly
    annotator_output_dir = os.path.join(OUTPUT_DIR, ANNOTATOR_OUTPUT_DIR)

    if os.path.exists(annotator_output_dir):
        raise FileExistsError(f"The directory '{annotator_output_dir}' already exists.")
    os.makedirs(annotator_output_dir)
    annotator_input_filepath = os.path.join(annotator_output_dir, "annotator_input.csv")
    response_dataset.to_annotator_input_csv(annotator_input_filepath)
    for a in config.safety_model_config.safety_models:
        command = [
            "poetry",
            "run",
            "modelgauge",
            "run-csv-items",
            "-a",
            a,
            annotator_input_filepath,
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running modelgauge for model {a}: {result.stderr}")
            exit(1)
        else:
            print(f"Successfully ran modelgauge for model {a}: {result.stdout}")

    # Join the annotation results into pseudo ground truths by looking in the dir
    annotations = []
    for filename in os.listdir(annotator_output_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(annotator_output_dir, filename)
            run = AnnotationRun.from_jsonl(filepath)

            # Check that all rows were generated (no silent failures... no way to tell other than double checking)
            if len(run) != len(response_dataset):
                raise ValueError(
                    f"Annotation count for {filename} does not match input dataset count. Annotation count: {len(run)}. Response dataset count: {len(response_dataset)}"
                )

            safety_model_annotation = run.to_safety_model_annotation()
            annotations.append(safety_model_annotation)

    # Create pseudo annotation joined table
    concat_annotation_records = SafetyModelAnnotationDataset.concat(annotations)
    concat_annotation_records.to_csv(os.path.join(OUTPUT_DIR, "annotations.csv"))
    pseudo_gt_dataset = concat_annotation_records.compute_pseudo_ground_truths()
    pseudo_gt_schema = PseudoGroundTruthDatasetSchema()
    result = response_dataset.df.merge(
        pseudo_gt_dataset.df, on=pseudo_gt_schema.pair_uid, how="left"
    )

    result.to_csv(
        os.path.join(OUTPUT_DIR, "pseudo_ground_truth_labels.csv"), index=False
    )


if __name__ == "__main__":
    main()
