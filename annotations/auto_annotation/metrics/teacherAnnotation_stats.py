import argparse
import json
import logging
import os
import subprocess

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

### Utility functions


#### Save to csv
def save_to_csv(data, columns, file_path):
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)


#### Function to compute percentages and counts
def compute_counts_and_percentages(data):
    counts = data.value_counts()
    percentages = counts / len(data) * 100
    return pd.DataFrame({"count": counts, "percentage": percentages})


#### Writing statistics to file
def write_stats_to_file(stats, total_samples, filename, header):
    with open(filename, "a") as f:
        f.write(f"{header}\n")
        f.write(f"{stats.to_string()}\n")
        f.write(f"Total samples: {total_samples}\n\n")


#### Running teacher annoation command
def run_command(command, file_path):
    try:
        full_command = f"{command} {file_path}"
        result = subprocess.run(
            full_command, shell=True, check=True, capture_output=True, text=True
        )
        logging.info(
            f"Command executed successfully on {file_path}\nOutput: {result.stdout}"
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Error executing command on {file_path}: {e}\nError output: {e.stderr}"
        )


def run_annotators(file_path, annotators):
    for ann in annotators:
        run_command(f"poetry run modelgauge run-annotators -a {ann}", file_path)


#### Given prompts, modelgauge generates responses from various SUTs. This creates a json file with
#### the prompt, SUT response along with all the other initial details such as hazard type, persona, etc..
#### This function processes the files and creates 2 tables/records, one containing all the detailed info,
#### And the other containing only the UID, Prompt, Response,SUT - as required for the next step, i.e. Teacher annotation.
def process_modelgauge_SUTResponse_json(file_path, sut):
    with open(file_path, "r") as f:
        records = json.load(f)["test_item_records"]

    table_data = []
    table_data_teacher = []

    for record in records:
        prompt = record["test_item"]["prompts"][0]["prompt"]
        response = record["interactions"][0]["response"]["completions"][0]["completion"]
        context = record["test_item"]["context_internal"]["data"]
        uid = record["test_item"]["prompts"][0]["source_id"]

        row = [
            uid,  # UID
            context["persona_type"],
            context["hazard"],
            prompt["text"],  # Prompt
            response["text"],  # Response
            sut,  # SUT
            prompt["options"]["temperature"],
            prompt["options"]["top_p"],
        ]
        table_data.append(row)
        ### Write only columns required for running teacher annotations
        table_data_teacher.append(
            [row[0], row[3], row[4], row[5]]
        )  # UID, Prompt, Response, SUT

    return table_data, table_data_teacher


### This function processes the teacher annotation output file
### Each json has the Prompt, Response, SUT along with the teacher annotation output (_is_safe) + associated metadata
def read_jsonl_teacherannotation_to_df(file_path):
    data = []
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, 1):
            try:
                json_object = json.loads(line.strip())
                data.append(json_object)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON on line {line_number}: {e}")
                logging.error(
                    f"Problematic line: {line[:100]}..."
                )  # Log first 100 chars of the line

    if not data:
        logging.warning("No valid JSON objects found in the file.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    df.columns = df.columns.str.replace(".", "_", regex=False)

    df_unique = df.drop_duplicates(subset=["UID"], keep="first")
    print("Original Teacher Annotation Output Length = ", len(df))
    print("After removing duplicates, length = ", len(df_unique))
    return df_unique


# ### Compute unianimous safe and unsafe outputs combining all teacher outputs
def get_safe_result(row):
    is_safe_columns = [col for col in row.index if col.endswith("_is_safe")]
    # print("is_safe_columns = ", is_safe_columns)
    safe = sum(row[col] == True for col in is_safe_columns)
    unsafe = sum(row[col] == False for col in is_safe_columns)

    if unsafe == len(is_safe_columns):
        return "unanimous_unsafe"
    elif safe == len(is_safe_columns):
        return "unanimous_safe"
    elif safe == 2 and unsafe == 2:
        return "max_disagreement"
    elif safe == 3 and unsafe == 1:
        return "majority_safe"
    elif safe == 1 and unsafe == 3:
        return "majority_unsafe"
    else:
        return "No conclusion"


def compute_consensus_metrics(input_df):
    safety_cols = [col for col in input_df.columns if col.endswith("_is_safe")]
    print("safety cols are:", safety_cols)
    for col in safety_cols:
        input_df[col] = input_df[col].astype(int)

    # Inter-rater reliability measure - fleiss kappa
    ratings_for_fleiss = prepare_ratings_for_fleiss(input_df, safety_cols)
    fleiss = fleiss_kappa(ratings_for_fleiss)

    # Model comparison (permissiveness)
    model_permissiveness = input_df[safety_cols].mean().sort_values(ascending=False)
    return fleiss, model_permissiveness  # , input_df


def prepare_ratings_for_fleiss(df, safety_cols):
    # Convert ratings to format required by fleiss_kappa function
    n_categories = 2  # Safe (1) and Unsafe (0)
    ratings = np.zeros((len(df), n_categories))
    for col in safety_cols:
        ratings[:, 1] += df[col]  # Count of "Safe" ratings
    ratings[:, 0] = len(safety_cols) - ratings[:, 1]  # Count of "Unsafe" ratings
    return ratings


def combine_allannotations_orig_data(
    merged_df, orig_prompts_file, output_dir, prefix, stats_filename
):
    merged_df.to_csv(os.path.join(output_dir, prefix + "_mergedteacherOutputsOnly.csv"))
    print(merged_df.columns)
    orig_prompts = pd.read_csv(orig_prompts_file)
    allcols_df = pd.merge(
        orig_prompts, merged_df, on="UID"
    )  # left_on="prompt_uid", right_on="UID")
    print("LEN(ALCOLS_DF) = ", len(allcols_df))
    # all_df_votes = get_agreement_disagreement(allcols_df)
    allcols_df["safe_unsafe_output"] = allcols_df.apply(get_safe_result, axis=1)
    allcols_df.to_csv(
        os.path.join(output_dir, prefix + "_CombinedPromptsTeacherAnnotations.csv"),
        index=False,
    )

    ### Filter out skilled, unskilled:

    for persona in ["skilled", "unskilled"]:
        persona_df = allcols_df[allcols_df["persona"] == persona]
        results = pd.Series(persona_df["safe_unsafe_output"].tolist())
        stats = compute_counts_and_percentages(results)
        write_stats_to_file(
            stats, len(persona_df), stats_filename, f"{persona.capitalize()} Results:"
        )
        persona_df.to_csv(
            os.path.join(output_dir, prefix + "_" + persona + ".csv"), index=False
        )

    #### Filter out unanimous safes & unsafes, majority safes & unsafes, and max_disagreement and output to csv.
    for output_type in [
        "unanimous_safe",
        "unanimous_unsafe",
        "majority_safe",
        "majority_unsafe",
        "max_disagreement",
    ]:
        result_df = allcols_df[allcols_df["safe_unsafe_output"] == output_type]
        result_df.to_csv(
            os.path.join(output_dir, prefix + "_" + output_type + ".csv"), index=False
        )

    all_results = pd.Series(allcols_df["safe_unsafe_output"].tolist())
    overall_stats = compute_counts_and_percentages(all_results)
    write_stats_to_file(
        overall_stats,
        len(all_results),
        stats_filename,
        f"Overall Results for " + output_dir + ":",
    )

    return allcols_df
