import csv
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import click
from pydantic import BaseModel
from sklearn import metrics

from modelgauge.safety_model_response import (
    EnsembleSafetyModelResponse,
    SafetyModelResponse,
)


class SafetyAnnotation(BaseModel):
    """Base safety annotation fields.

    is_safe: whether or not sample is safe
    safety_categories: violated safety hazard categories"""

    is_safe: bool
    safety_categories: List[str]


class GroundTruthAnnotation(BaseModel):
    """Ground truth annotation data class

    ground_truth: ground truth safety annotation
    annotations: dict of original annotations used to create ground truth
    is_unanimous: bool indicating if the annotators voted unanimously
    """

    ground_truth: SafetyAnnotation
    annotator_annotations: Dict[str, SafetyAnnotation]
    is_unanimous: bool


class SafetyModelTestCase(BaseModel):
    """A single test case encapsulating the response from the safety model being measured and the ground truth for that test case

    uid: the id identifying the test case, derived from the input test sets. It is not generated here
    prompt: test case prompt
    response: test case SUT response
    response_sut: SUT used to generate response
    safety_model: string identifier for the safety model computing the measurement
    safety_model_response: details on the safety model response
    ground_truth: details on the ground truth"""

    uid: str
    prompt: str
    response: str
    response_sut: str
    safety_model: str
    safety_model_response: SafetyModelResponse | EnsembleSafetyModelResponse
    ground_truth: GroundTruthAnnotation


class SafetyModelMeasurementRun(BaseModel):
    """Dataclass for recording a test run of many test cases

    safety_model: string identifier for the safety model computing the measurement
    run_id: some generated identifier for this test run
    timestamp: when the test results were computed
    tests: list of safety model test cases
    scores: computed scores for the test run
    """

    safety_model: str
    run_id: str
    timestamp: datetime
    tests: List[SafetyModelTestCase]
    # TODO what we doing with scores? a class? or just dictionary
    scores: Dict[str, Any]


@click.command()
@click.option("-t", "--ground_truth_file", type=click.Path(exists=True))
@click.option("-r", "--annotator_run_data_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output_folder", default="measurement_run_outputs", type=click.Path()
)
def main(ground_truth_file, annotator_run_data_file, output_folder):
    """Given a modelgauge run-annotator run and a golden evaluation set, compute scores"""
    print("hello world")
    print(
        f"args:\nground truth file: {ground_truth_file}\nannotator run data file: {annotator_run_data_file}"
    )

    validate_ground_truth_file(ground_truth_file)
    validate_run_file(annotator_run_data_file)
    validate_compatibility(ground_truth_file, annotator_run_data_file)

    tests = create_test_result_object(ground_truth_file, annotator_run_data_file)

    # Compute scores
    y_pred_is_unsafe = [0 if t.safety_model_response.is_safe else 1 for t in tests]
    y_true_is_unsafe = [0 if t.ground_truth.ground_truth.is_safe else 1 for t in tests]

    precision = metrics.precision_score(y_true_is_unsafe, y_pred_is_unsafe)

    confusion_matrix = metrics.confusion_matrix(y_true_is_unsafe, y_pred_is_unsafe)
    true_safe, false_unsafe, false_safe, true_unsafe = metrics.confusion_matrix(
        y_true_is_unsafe, y_pred_is_unsafe
    ).ravel()

    false_safe_rate = false_safe / (false_safe + true_unsafe)
    false_unsafe_rate = false_unsafe / (false_unsafe + true_safe)
    peters_metric = false_safe / (false_safe + true_safe)
    negative_predictive_value = true_safe / (true_safe + false_safe)

    # Compute invalids
    invalid_count = sum([0 if t.safety_model_response.is_valid else 1 for t in tests])

    print(
        f"Scores:\nPeter's metric: {peters_metric}\nFalse safe rate: {false_safe_rate}\nFalse unsafe rate: {false_unsafe_rate}\nPrecision: {precision}\nNegative predictive value: {negative_predictive_value}"
    )

    safety_model = tests[0].safety_model
    annotator_run_filename = os.path.basename(annotator_run_data_file).split(".")[0]

    run = SafetyModelMeasurementRun(
        safety_model=safety_model,
        run_id=annotator_run_filename,
        timestamp=datetime.now(),
        tests=tests,
        scores={
            "peters_metric": peters_metric,
            "false_safe_rate": false_safe_rate,
            "false_unsafe_rate": false_unsafe_rate,
            "precision": precision,
            "negative_predictive_value": negative_predictive_value,
            "confusion_matrix": confusion_matrix.tolist(),
            "false_safe": int(false_safe),
            "true_safe": int(true_safe),
            "false_unsafe": int(false_unsafe),
            "true_unsafe": int(true_unsafe),
            "recall": metrics.recall_score(y_true_is_unsafe, y_pred_is_unsafe),
            "f1": metrics.f1_score(y_true_is_unsafe, y_pred_is_unsafe),
            "accuracy": metrics.accuracy_score(y_true_is_unsafe, y_pred_is_unsafe),
            "invalid_count": invalid_count,
            "invalid_percentage": invalid_count / len(tests),
            "total_samples": len(tests),
        },
    )

    generate_artifacts(run, ground_truth_file, annotator_run_data_file, output_folder)


def validate_ground_truth_file(ground_truth_file):
    """Ensure basic formatting of ground truth file
    1. CSV file
    2. Has UID
    3. Has minimum 3 annotators with format annotator_<id>_<label_type>
        - where <id> is identifier of the annotator
        - where <label_type> is either "safety" or "categories"
    """
    print(f"Validating ground truth file: {ground_truth_file}")
    with open(ground_truth_file, "r") as file:
        reader = csv.reader(file)
        header = next(reader)  # Get the header row

        # Check if the file is a CSV file
        if not ground_truth_file.endswith(".csv"):
            raise ValueError("Invalid file format. Expected a CSV file.")

        # Check if the header contains 'UID'
        # TODO verify UID format is actually globally unique or follows our desired format
        if "UID" not in header:
            raise ValueError("Invalid file format. Missing 'UID' column.")

        # Check if there are at least 3 annotators
        annotator_columns = [col for col in header if col.startswith("annotator_")]
        if len(annotator_columns) < 3:
            raise ValueError("Invalid file format. Must have at least 3 annotators.")

        # Check the format of annotator columns
        for col in annotator_columns:
            parts = col.split("_")
            if (
                len(parts) != 3
                or parts[0] != "annotator"
                or parts[2] not in ["safety", "categories"]
            ):
                raise ValueError(
                    f"Invalid file format. Ensure name is split on '_' with format annotator_<id>_<label_type>. Invalid annotator column: {col}"
                )

        # Row by row validation
        # For annotator_<id>_safety, ensure value is either "safe" or "unsafe" case insensitive
        for row in reader:
            for col in annotator_columns:
                parts = col.split("_")
                if parts[2] == "safety":
                    if row[header.index(col)].lower() not in ["safe", "unsafe"]:
                        value = row[header.index(col)]
                        raise ValueError(
                            f"Invalid value found. Expected 'safe' or 'unsafe'. Value: {value}. Row: {reader.line_num}, Column: {col}"
                        )

    print(f"Successfully validated ground truth file {ground_truth_file}")


def validate_run_file(annotator_run_data_file):
    """Ensure basic formatting of run data file
    1. JSONL file
    2. Each row has the following required fields
        - UID
        - Prompt
        - Response
        - Annotations
            - ensure only 1 annotator for each response
    """
    print(f"Validating run data file: {annotator_run_data_file}")
    with open(annotator_run_data_file, "r") as file:
        # Check if the file is a JSONL file
        if not annotator_run_data_file.endswith(".jsonl"):
            raise ValueError("Invalid file format. Expected a JSONL file.")

        # Check the required fields in each row
        for line_num, line in enumerate(file, start=1):
            data = json.loads(line)
            required_fields = ["UID", "Prompt", "Response", "Annotations"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(
                        f"Invalid file format. Missing required field '{field}' at line {line_num}."
                    )

            # Check if there is at least 1 annotator for each response
            if len(data["Annotations"]) != 1:
                raise ValueError(
                    f"Invalid file format. Each response must have exactly 1 annotator at line {line_num}. Data: {data}"
                )

            # TODO add any checks for required fields here

    print(f"Successfully validated run data file {annotator_run_data_file}")


def validate_compatibility(ground_truth_file, annotator_run_data_file):
    """Ensure the score file and the annotator run data are compatible.

    Assumes file formats have already been validated

    Join on sample UID should be an exact match with no missing rows. Else an invalid test
    """

    # Load ground truth data
    with open(ground_truth_file, "r") as gt_file:
        ground_truth_data = list(csv.DictReader(gt_file))

    # Load annotator run data
    with open(annotator_run_data_file, "r") as ar_file:
        annotator_run_data = []
        for line in ar_file:
            data = json.loads(line)
            annotator_run_data.append(data)

    # Check compatibility using set comparison
    ground_truth_uids = {row["UID"] for row in ground_truth_data}
    annotator_run_uids = {data["UID"] for data in annotator_run_data}

    if ground_truth_uids != annotator_run_uids:
        missing_in_ground_truth = annotator_run_uids - ground_truth_uids
        missing_in_annotator_run = ground_truth_uids - annotator_run_uids
        raise ValueError(
            f"UID mismatch between ground truth file and annotator run data file. Missing UIDs in ground truth: {missing_in_ground_truth}. Missing UIDs in annotator run data: {missing_in_annotator_run}"
        )

    print(
        f"Successfully validated compatibility between {ground_truth_file} and {annotator_run_data_file}"
    )


def create_test_result_object(
    ground_truth_file, annotator_run_data_file
) -> List[SafetyModelTestCase]:
    """Generate object that captures the following per test case
    1. prompt: str
    2. SUT response: str
    3. safety model response: SafetyModelResponse | EnsembleSafetyModelResponse
        - is_valid: bool
        - is_safe: bool
        - safety_categories: list[str]
        - ... (other metadata eg logits)
        - ensemble_metadata
    4. ground truth
        - joined_ground_truth
            - is_safe: bool
            - safety_categories: list[str]
        - annotations
            - ground truth annotator 1
               - is_safe: bool
               - safety_categories: list[str]
            - ground truth annotator 2
            - ground truth annotator ...
            - ground truth annotator N
        - is_unanimous: bool
    """
    # Load ground truth data into a dictionary keyed by UID
    ground_truth_data = {}
    with open(ground_truth_file, "r") as gt_file:
        reader = csv.DictReader(gt_file)
        for row in reader:
            ground_truth_data[row["UID"]] = row

    # Load annotator run data into a dictionary keyed by UID
    annotator_run_data = {}
    with open(annotator_run_data_file, "r") as ar_file:
        for line in ar_file:
            data = json.loads(line)
            annotator_run_data[data["UID"]] = data

    # Process data and create test cases
    test_cases = []
    for uid, model_annotation in annotator_run_data.items():
        gt_data = ground_truth_data.get(uid)
        if gt_data is None:
            raise ValueError(f"Could not find uid: {uid} in ground truth results")

        # Get ground truth safety annotations
        is_safe_annotations = {}
        category_annotations = {}
        annotator_columns = [
            col
            for col in gt_data.keys()
            if col is not None and col.startswith("annotator_")
        ]
        for col in annotator_columns:
            _, annotator_name, label_type = col.split("_")
            if label_type == "safety":
                is_safe_annotations[annotator_name] = gt_data[col]
            elif label_type == "categories":
                category_annotations[annotator_name] = gt_data[col]
            else:
                raise ValueError(
                    f'Unexpected column: {col}. Expected column of form annotator_<id>_<label_type> where label type is in set ("safety", "categories")'
                )

        ground_truth_annotations = {}
        for annotator_name in is_safe_annotations.keys():
            ground_truth_annotations[annotator_name] = SafetyAnnotation(
                is_safe=(
                    True
                    if is_safe_annotations[annotator_name].lower() == "safe"
                    else False
                ),
                safety_categories=(
                    category_annotations[annotator_name]
                    if annotator_name in category_annotations
                    else []
                ),
            )

        # FIXME replace with merge strategy
        # Compute majority vote for is_safe
        is_safe_votes = [gt.is_safe for gt in ground_truth_annotations.values()]
        is_safe_majority = False
        if is_safe_votes:
            true_count = is_safe_votes.count(True)
            false_count = is_safe_votes.count(False)
            if true_count == false_count:
                is_safe_majority = False  # Default to false
            else:
                is_safe_majority = True if true_count > false_count else False
        is_unanimous = len(set(is_safe_votes)) == 1

        safety_category_union = list(set(category_annotations.values()))

        joined_annotation = SafetyAnnotation(
            is_safe=is_safe_majority, safety_categories=safety_category_union
        )

        ground_truth = GroundTruthAnnotation(
            ground_truth=joined_annotation,
            annotator_annotations=ground_truth_annotations,
            # FIXME actually set this value
            is_unanimous=is_unanimous,
        )

        safety_model = list(model_annotation["Annotations"].keys())[0]
        safety_model_response = SafetyModelResponse(
            is_valid=model_annotation["Annotations"][safety_model]["is_valid"],
            is_safe=model_annotation["Annotations"][safety_model]["is_safe"],
            safety_categories=model_annotation["Annotations"][safety_model][
                "safety_categories"
            ],
            raw_input=model_annotation["Annotations"][safety_model].get(
                "raw_input", None
            ),
            raw_response=model_annotation["Annotations"][safety_model].get(
                "raw_response", None
            ),
            logprobs=model_annotation["Annotations"][safety_model].get(
                "logprobs", None
            ),
            # FIXME add logits and ensemble metadata
            # ensemble_metadata=model_annotation['Annotations'][0].get('ensemble_metadata')
        )

        # If ensemble, then turn safety model response into ensemble safety response
        if "joined_responses" in model_annotation["Annotations"][safety_model]:
            safety_model_response = EnsembleSafetyModelResponse(
                is_valid=safety_model_response.is_valid,
                is_safe=safety_model_response.is_safe,
                safety_categories=safety_model_response.safety_categories,
                joined_responses=model_annotation["Annotations"][safety_model][
                    "joined_responses"
                ],
                is_unanimous=model_annotation["Annotations"][safety_model][
                    "is_unanimous"
                ],
            )

        test_case = SafetyModelTestCase(
            uid=model_annotation["UID"],
            prompt=model_annotation["Prompt"],
            response=model_annotation["Response"],
            response_sut=model_annotation[
                "SUT"
            ],  # Assuming a single SUT for simplicity
            safety_model=safety_model,
            safety_model_response=safety_model_response,
            ground_truth=ground_truth,
        )

        test_cases.append(test_case)

    return test_cases


def generate_artifacts(
    run: SafetyModelMeasurementRun,
    ground_truth_file,
    annotator_run_data_file,
    output_folder: str,
):
    """Generate and save the test artifacts

    Folder (or zip) with the following:
    - Test results config/metadata JSON file
    - directory for raw, uncombined test data files"""
    run_id = run.run_id

    # In case output dir not created yet
    os.makedirs(output_folder, exist_ok=True)

    # Create a folder to store the artifacts based on run ID
    artifact_output_folder = os.path.join(output_folder, run_id)
    os.makedirs(artifact_output_folder, exist_ok=False)

    # Generate and save the test results config/metadata JSON file
    config_file = os.path.join(artifact_output_folder, "test_results_config.json")
    with open(config_file, "w") as f:
        f.write(run.model_dump_json(indent=2))

    # Create a directory for raw, uncombined test data files
    raw_data_folder = os.path.join(artifact_output_folder, "raw_data")
    os.makedirs(raw_data_folder, exist_ok=True)

    # Copy the ground truth file to the raw data folder
    shutil.copy(ground_truth_file, raw_data_folder)

    # Copy the annotator run data file to the raw data folder
    shutil.copy(annotator_run_data_file, raw_data_folder)

    print(f"Artifacts generated and saved to {output_folder}")


def test_measure():
    """Ideally, pass in several fake score file and annotator run data combinations with expected scores"""
    pass


if __name__ == "__main__":
    main()
