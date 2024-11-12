import argparse
import os
import shutil
import subprocess
from datetime import datetime
from typing import Optional

import yaml
from pydantic import BaseModel

from modelgauge.ensemble.composer import EnsembleComposer
from runners.safety_model_config import SafetyModelConfig
from schemas.safety_datasets import (
    AnnotationInputDataset,
    AnnotationRun,
    ResponseDataset,
)

SAFETY_MODEL_OUTPUTS = "safety_model_outputs"
ANNOTATOR_TEMP_DIR = "annotator_temp_dir"
SAMPLE_CONFIG_PATH = "./sample_resources/sample_pseudo_gt_config.yml"

INCOMPLETE_ANNOTATOR_TEMP_DIR = "incomplete_annotator_temp_dir"
INCOMPLETE_OUTPUTS_TEMP_DIR = "incomplete_annotator_outputs_temp_dir"

ANNOTATOR_INPUT_FNAME = "annotator_input.csv"


class SafetyModelsRunnerConfig(BaseModel):
    responses_dataset_path: str
    safety_model_config: SafetyModelConfig

    @classmethod
    def from_yml(cls, filepath: str):
        with open(filepath, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


def main():
    parser = argparse.ArgumentParser(
        description="Run safety model annotations on a specified dataset"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    print("Loading config...")
    config = SafetyModelsRunnerConfig.from_yml(args.config)
    print("Done loading config")

    run_safety_models(config)


def run_safety_models(
    config: SafetyModelsRunnerConfig, run_output_dir: Optional[str] = None
):
    print("Setting up working directories...")

    # Load SUT responses
    response_dataset = ResponseDataset.from_csv(config.responses_dataset_path)

    # Create run output dir
    if run_output_dir:
        if os.path.exists(run_output_dir):
            raise FileExistsError(f"The directory '{run_output_dir}' already exists.")
        os.makedirs(run_output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = f"./run_{timestamp}"
        os.makedirs(run_output_dir)

    # Create temporary working output dir for modelgauge commands
    annotator_temp_working_dir = os.path.join(run_output_dir, ANNOTATOR_TEMP_DIR)
    if os.path.exists(annotator_temp_working_dir):
        raise FileExistsError(
            f"The directory '{annotator_temp_working_dir}' already exists."
        )
    os.makedirs(annotator_temp_working_dir)

    # Create the safety model output dir for safety model annotation outputs
    safety_model_output_dir = os.path.join(run_output_dir, SAFETY_MODEL_OUTPUTS)
    if os.path.exists(safety_model_output_dir):
        raise FileExistsError(
            f"The directory '{safety_model_output_dir}' already exists."
        )
    os.makedirs(safety_model_output_dir)

    # Transform the response dataset into the intermediate formate needed for
    # the annotators
    annotation_input_filepath = os.path.join(
        annotator_temp_working_dir, ANNOTATOR_INPUT_FNAME
    )
    annotation_input_dataset = response_dataset.to_annotation_dataset()
    annotation_input_dataset.to_csv(annotation_input_filepath)

    safety_model_output_filepaths = {}

    # Track which safety model IDs we ran so we don't need to run them multiple times
    completed_safety_model_runs = set()

    # Check if there are completed runfiles already
    for safety_model in config.safety_model_config.safety_models:
        if safety_model.runfile:
            print(
                f"Config specifies a runfile to use instead of regenerating: {safety_model.runfile}"
            )

            if not os.path.exists(safety_model.runfile):
                raise FileNotFoundError(
                    f"Runfile '{safety_model.runfile}' does not exist."
                )

            if safety_model.runfile_is_complete(annotation_input_dataset):
                print(
                    f"Specified runfile for safety model: '{safety_model.name}' is complete. It has annotations for all UIDs in input dataset."
                )
                runfile_copy_destination = get_safety_model_output_filepath(
                    safety_model.name, safety_model_output_dir
                )
                shutil.copy(safety_model.runfile, runfile_copy_destination)
                print(
                    f"Successfully loaded runfile: Copied {safety_model.runfile} to {runfile_copy_destination}"
                )

                completed_safety_model_runs.add(safety_model.name)
                safety_model_output_filepaths[safety_model.name] = (
                    runfile_copy_destination
                )
            else:
                if not safety_model.runfile_uids_in_input_dataset(
                    annotation_input_dataset
                ):
                    raise ValueError(
                        f"Runfile provided for safety model '{safety_model.name}' is incompatible with the input dataset. Runfile {safety_model.runfile} has UIDs not in the input dataset."
                    )
                else:
                    missing_uids = safety_model.get_missing_samples_from_runfile(
                        annotation_input_dataset
                    )
                    if safety_model.ensemble:
                        raise ValueError(
                            f"Runfile provided for ensemble safety model '{safety_model.name}' is incompatible with the input dataset. Runfile {safety_model.runfile} is missing {len(missing_uids)} UIDs from the input dataset."
                        )
                    else:
                        print(
                            f"Runfile for safety model '{safety_model.name}' is partially completed. Will resume later."
                        )

    # Get responses for models
    for safety_model in config.safety_model_config.safety_models:
        print(f"Starting run for safety model: {safety_model.name}")

        if safety_model.ensemble:
            print("Starting ensemble run.")
            response_files_to_join = []

            # Get annotations for each member
            for ensemble_member in safety_model.ensemble.safety_models:
                print(f"Running for ensemble member: {ensemble_member}")
                if ensemble_member in completed_safety_model_runs:
                    print(f"Already ran safety model: {ensemble_member}. Skipping...")
                    response_files_to_join.append(
                        get_safety_model_output_filepath(
                            ensemble_member, safety_model_output_dir
                        )
                    )
                else:
                    response_filepath = get_safety_model_output_filepath(
                        ensemble_member, safety_model_output_dir
                    )
                    run_modelgauge_annotations_command(
                        ensemble_member,
                        annotator_temp_working_dir,
                        annotation_input_dataset,
                        annotation_input_filepath,
                        ANNOTATOR_INPUT_FNAME,
                        response_filepath,
                    )
                    completed_safety_model_runs.add(ensemble_member)
                    response_files_to_join.append(response_filepath)
                    safety_model_output_filepaths[ensemble_member] = response_filepath

            # Combine ensemble member responses into ensemble response file
            print(
                f"Combining ensemble results for {safety_model.name} using response files: {response_files_to_join}"
            )
            ensemble_response_filepath = get_safety_model_output_filepath(
                safety_model.name, safety_model_output_dir
            )
            ensemble_composer = EnsembleComposer(
                ensemble_strategy=safety_model.ensemble.get_join_strategy()
            )
            ensemble_composer.compose_responses_to_file(
                new_ensemble_id=safety_model.name,
                file_paths=response_files_to_join,
                output_file_path=ensemble_response_filepath,
            )
            safety_model_output_filepaths[safety_model.name] = (
                ensemble_response_filepath
            )

        else:
            if safety_model.name in completed_safety_model_runs:
                print(f"Already ran safety model: {safety_model.name}. Skipping...")
                continue

            response_filepath = get_safety_model_output_filepath(
                safety_model.name, safety_model_output_dir
            )

            if safety_model.runfile and not safety_model.runfile_is_complete(
                annotation_input_dataset
            ):
                missing_uids = safety_model.get_missing_samples_from_runfile(
                    annotation_input_dataset
                )
                run_command_for_missing_uids(
                    missing_uids=list(missing_uids),
                    partial_run=AnnotationRun.from_jsonl(safety_model.runfile),
                    annotation_input_dataset=annotation_input_dataset,
                    annotator_id=safety_model.name,
                    parent_run_output_dir=run_output_dir,
                    safety_model_output_dir=safety_model_output_dir,
                )
            else:
                response_filepath = get_safety_model_output_filepath(
                    safety_model.name, safety_model_output_dir
                )
                run_modelgauge_annotations_command(
                    safety_model.name,
                    annotator_temp_working_dir,
                    annotation_input_dataset,
                    annotation_input_filepath,
                    ANNOTATOR_INPUT_FNAME,
                    response_filepath,
                )

            completed_safety_model_runs.add(safety_model.name)
            safety_model_output_filepaths[safety_model.name] = response_filepath

    print("Finished running all safety models")

    # Return mappings from safety model names to their safety model output locations
    return safety_model_output_filepaths


def get_safety_model_output_filepath(annotator_id: str, safety_model_output_dir: str):
    output_filename = f"{annotator_id}_annotations.jsonl"
    output_filepath = os.path.join(safety_model_output_dir, output_filename)
    return os.path.abspath(output_filepath)


def run_modelgauge_annotations_command(
    annotator_id: str,
    temp_working_dir: str,
    annotation_input_dataset: AnnotationInputDataset,
    annotation_input_filepath: str,
    annotation_filename: str,
    output_filepath: str,
):
    """Runs the modelgauge command and collects the output file.

    Some of the logic here is moving between a temporary output directory and a
    named output file which can be removed if modelgauge adds the ability to
    specify output file locations (it currently does not)

    annotator_id: name of the annotator in modelguage
    temp_working_dir: temporary working directory where intermediate outputs and generated and where the annotation input file is
    """
    if not os.path.exists(annotation_input_filepath):
        raise FileNotFoundError(
            f"Annotation input filepath '{annotation_input_filepath}' does not exist."
        )

    command = [
        "poetry",
        "run",
        "modelgauge",
        "run-csv-items",
        "-a",
        annotator_id,
        annotation_input_filepath,
    ]
    result = subprocess.run(command, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Error running modelgauge for model {annotator_id}: {result.stderr}")
        exit(1)
    else:
        print(f"Successfully ran modelgauge for model {annotator_id}: {result.stdout}")

    # Find, rename, and move the file to the safety model output directory
    temp_dir_files = os.listdir(temp_working_dir)
    temp_output_file = None

    # Because we don't know the name of the output file deterministically, we need to seek for it
    for file in temp_dir_files:
        if file != annotation_filename:
            temp_output_file = os.path.join(temp_working_dir, file)
            break

    if len(temp_dir_files) != 2:
        raise ValueError(
            f"Temporary output directory in unknown state. Expected 2 files: annotation input csv and the annotation output jsonl. Directory contents: {temp_dir_files}"
        )
    if temp_output_file == None:
        raise ValueError(
            f"Could not find annotation output jsonl file in the temporary output directory. Expected 2 files: annotation input csv and annotation output jsonl. Directory contents: {temp_dir_files}"
        )

    if os.path.exists(temp_output_file):
        shutil.move(temp_output_file, output_filepath)
        print(f"Moved {temp_output_file} to {output_filepath}")
    else:
        raise ValueError(f"Expected output file {temp_output_file} not found.")

    # Verify that the command generated a file with expected number of outputs
    run = AnnotationRun.from_jsonl(output_filepath)

    if len(run) != len(annotation_input_dataset):
        raise ValueError(
            f"Expected {len(annotation_input_dataset)} annotations, but got {len(run)}"
        )

    return output_filepath


def run_command_for_missing_uids(
    missing_uids: list[str],
    partial_run: AnnotationRun,
    annotation_input_dataset: AnnotationInputDataset,
    annotator_id: str,
    parent_run_output_dir: str,
    safety_model_output_dir: str,
):
    # Create temporary working dir for modelgauge commands
    working_dir = os.path.join(parent_run_output_dir, INCOMPLETE_ANNOTATOR_TEMP_DIR)
    os.makedirs(working_dir, exist_ok=True)

    # Create temporary missing uid output dir
    working_output_dir = os.path.join(
        parent_run_output_dir, INCOMPLETE_OUTPUTS_TEMP_DIR
    )
    os.makedirs(working_output_dir, exist_ok=True)

    # Create a new input dataset distilled from the old one
    missing_uids_input_dataset_filepath = os.path.join(
        working_dir, ANNOTATOR_INPUT_FNAME
    )
    missing_uids_input_dataset = annotation_input_dataset.get_subset(missing_uids)
    missing_uids_input_dataset.to_csv(missing_uids_input_dataset_filepath)

    # Run modelgauge on the missing dataset
    missing_uids_response_filepath = get_safety_model_output_filepath(
        annotator_id, working_output_dir
    )
    run_modelgauge_annotations_command(
        annotator_id=annotator_id,
        temp_working_dir=working_dir,
        annotation_input_dataset=missing_uids_input_dataset,
        annotation_input_filepath=missing_uids_input_dataset_filepath,
        annotation_filename=ANNOTATOR_INPUT_FNAME,
        output_filepath=missing_uids_response_filepath,
    )

    # Load response from output filepath
    missing_uids_completed_run = AnnotationRun.from_jsonl(
        missing_uids_response_filepath
    )

    # Merge
    merged_run = partial_run.merge(missing_uids_completed_run)

    if not merged_run.matches_annotation_input_dataset(annotation_input_dataset):
        raise ValueError(
            f"Merged run does not match the annotation input dataset for annotator {annotator_id}."
        )

    # Write merged output
    completed_response_filepath = get_safety_model_output_filepath(
        annotator_id, safety_model_output_dir
    )
    merged_run.to_jsonl(completed_response_filepath)

    return completed_response_filepath


if __name__ == "__main__":
    main()
