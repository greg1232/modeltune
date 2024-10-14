import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from teacherAnnotation_stats import *


def find_dups(table_data, cols):
    uids = [row[0] for row in table_data]  # Assuming UID is the first column
    uid_counts = Counter(uids)

    duplicate_uids = [uid for uid, count in uid_counts.items() if count > 1]

    df = pd.DataFrame(table_data, columns=cols)

    df_unique = df.drop_duplicates(subset=[cols[0]], keep="first")
    unique_data = df_unique.values.tolist()
    return unique_data


def process_sut_responseFile(file_to_read, teacherAnnotationDir, sut, prefix):

    table_data, table_data_teacher = process_modelgauge_SUTResponse_json(
        file_to_read, sut
    )
    sutResponsesCsvFile = os.path.join(teacherAnnotationDir, prefix + ".csv")
    uniq_table_data = find_dups(
        table_data,
        [
            "UID",
            "persona",
            "Hazard",
            "Prompt",
            "Response",
            "SUT",
            "temperature",
            "top_p",
        ],
    )
    uniq_table_data_teacher = find_dups(
        table_data_teacher, ["UID", "Prompt", "Response", "SUT"]
    )

    # print("len(table_data = ", len(table_data), "\tlen(uniq tabe data = ", len(uniq_table_data))
    # print("len(table_data_teacher = ", len(table_data_teacher), "\tlen(uniq table_data_teacher = ", len(uniq_table_data_teacher))

    save_to_csv(
        uniq_table_data,
        [
            "UID",
            "persona",
            "Hazard",
            "Prompt",
            "Response",
            "SUT",
            "temperature",
            "top_p",
        ],
        sutResponsesCsvFile,
    )

    teacher_file = os.path.join(
        teacherAnnotationDir, prefix + "_forTeacherAnnotation.csv"
    )

    save_to_csv(
        uniq_table_data_teacher, ["UID", "Prompt", "Response", "SUT"], teacher_file
    )


def merge_teacherannotation_outputs(teacherAnnotationOutput_dir, prefix):
    merge_on_columns = ["UID", "Prompt", "SUT", "Response"]
    # merge_on_columns = ['UID']
    merged_df = None
    for teacherAnnotationFile in os.listdir(teacherAnnotationOutput_dir):
        if teacherAnnotationFile.startswith(prefix) and teacherAnnotationFile.endswith(
            ".jsonl"
        ):
            print("Processing Teacher Annotation File: ")
            print(teacherAnnotationFile)
            df = read_jsonl_teacherannotation_to_df(
                os.path.join(teacherAnnotationOutput_dir, teacherAnnotationFile)
            )
            if merged_df is None:
                merged_df = df
            else:
                print("MERGE DF COLUMNS = ", merged_df.columns)
                print("LEN(MERGED_DF) = ", len(merged_df), "LEN(DF)= ", len(df))
                merged_df = pd.merge(merged_df, df, on=merge_on_columns, how="inner")
    return merged_df


def readAndAnnotateSUTResponseFiles(
    sut_responses_dir, teacherAnnotationOutput_dir, output_dir
):
    print("Processing sut_responses Directory: ", sut_responses_dir)
    for eachfile in os.listdir(sut_responses_dir):
        if not eachfile.startswith(".") and eachfile.endswith(".json"):
            print("PROCESSING FILE: ", eachfile)
            prefix = eachfile.split(".json")[0]

            output_folder = os.path.join(output_dir, prefix)
            sut = eachfile.split("_")[-1].split(".json")[0]

            print("CREATING DIR:", output_folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            #### Process the SUT response file (json output from modelgauge)
            process_sut_responseFile(
                os.path.join(sut_responses_dir, eachfile), output_folder, sut, prefix
            )

            ###### Read the teacher annotation outputs for this SUT and combine all annotations
            merged_teacherAnnotations = merge_teacherannotation_outputs(
                teacherAnnotationOutput_dir, prefix
            )

            orig_prompts_file = os.path.join(output_folder, prefix + ".csv")
            stats_filename = os.path.join(output_folder, prefix + "_stats.txt")

            ###### Combine teacher annotations, along with initial prompt data
            ###### Split based on agreement/disagreement, persona, etc..
            allcols_df = combine_allannotations_orig_data(
                merged_teacherAnnotations,
                orig_prompts_file,
                output_folder,
                prefix,
                stats_filename,
            )

            results = compute_consensus_metrics(allcols_df)
            # fleiss_kappa, model_permissiveness, updated_df = results
            fleiss_kappa, model_permissiveness = results
            with open(stats_filename, "a") as f:
                f.write("-------------------\n\n")
                f.write("fleiss_kappa agreememt: \n")
                f.write(f"{fleiss_kappa}\n\n")

                f.write("model_permissiveness: \n")
                f.write(f"{model_permissiveness}\n\n")
                f.write("-------------------\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and analyze v1.0 practice runs"
    )
    parser.add_argument("--sut_responses_dir", help="Directory to process")
    parser.add_argument("--teacherAnnotationOutput_dir", help="Directory to process")
    parser.add_argument("--output_dir", help="Directory to process")
    args = parser.parse_args()

    readAndAnnotateSUTResponseFiles(
        args.sut_responses_dir, args.teacherAnnotationOutput_dir, args.output_dir
    )
