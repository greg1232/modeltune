import datetime
import pathlib
import subprocess
import re
from typing import List

import click

import mlflow
from modelgauge.config import load_secrets_from_config, raise_if_missing_from_config
from modelgauge.main import run_csv_items
from modelgauge.pipeline_runner import PromptRunner
from modelgauge.secret_values import MissingSecretValues, RawSecrets
from modelgauge.sut import SUT
from modelgauge.sut_capabilities import AcceptsTextPrompt
from modelgauge.sut_registry import SUTS


######################## I tried to extract the code from the run-csv-items command so I can add logs in between############################
######################## but it's not working yet. ##########################################################
def setup_credentials(sut_uids: list[str]) -> RawSecrets:
    """Load secrets from the config file and check for missing secrets."""
    # credentials
    secrets: RawSecrets = load_secrets_from_config()
    print("secrets:", secrets)
    # Check all objects for missing secrets.
    missing_secrets: List[MissingSecretValues] = []

    for sut_uid in sut_uids:
        print("sut_uid:", sut_uid)
        missing_secrets.extend(SUTS.get_missing_dependencies(sut_uid, secrets=secrets))

    raise_if_missing_from_config(missing_secrets)
    return secrets


def build_sut_instances(sut_uids: list[str], secrets: RawSecrets) -> dict[str, SUT]:
    """Create a dictionary of SUT instances."""
    suts = {}
    for sut_uid in sut_uids:
        sut = SUTS.make_instance(uid=sut_uid, secrets=secrets)
        if AcceptsTextPrompt not in sut.capabilities:
            raise click.BadParameter(f"{sut_uid} does not accept text prompts")
        suts[sut_uid] = sut
    return suts


def get_responses_from_SUTs(
    *,
    sut_uids: list[str],
    numb_worker_threads: int,
    input_file_path: str,
    cache_dir: str,
    debug: bool,
):
    with mlflow.start_run():
        # Log parameters
        params_to_be_logged = {
            "input_file_path": input_file_path,
            "numb_worker_threads": numb_worker_threads,
            "cache_dir": cache_dir,
            "sut_uids": sut_uids,
            "debug": debug,
        }
        mlflow.log_params(params_to_be_logged)

        # Step1: Setup credentials
        secrets: RawSecrets = setup_credentials(sut_uids=sut_uids)
        # Step 2: Build SUT instances
        sut_instances = build_sut_instances(sut_uids=sut_uids, secrets=secrets)

        # Step 3: Set up output file path
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file_path = input_file_path.parent / pathlib.Path(
            input_file_path.stem + "-responses-" + timestamp + ".csv"
        )
        mlflow.log_param("output_file_path", output_file_path)

        pipeline_runner = PromptRunner(
            num_workers=numb_worker_threads,
            input_path=input_file_path,
            output_path=output_file_path,
            cache_dir=cache_dir,
            suts=sut_instances,
        )

        pipeline_runner.run(progress_callback=None, debug=debug)
        mlflow.log_artifact(output_file_path)


def main():
    # input needed: sut_uids, annotator_uids, workers, cache_dir(for SUTs response), debug,
    sut_uids = ["llama-guard-3", "llama-3-70b-chat"]
    numb_worker_threads = len(sut_uids) * 10
    input_path = "mlflow/micro_prompt_uid.csv"
    cache_dir = "cached_responses"

    # get_responses_from_SUTs(
    #     sut_uids=sut_uids,
    #     numb_worker_threads=numb_worker_threads,
    #     input_file_path=input_path,
    #     cache_dir=cache_dir,
    #     debug=True
    # )
    mlflow.autolog()
    with mlflow.start_run():
        ############ I was able to make this run and I can see it on the MLFlow server. But no paras or artifacts is logged #################
        params_to_be_logged = {
            "input_file_path": input_path,
            "sut_uids": sut_uids,
        }
        mlflow.log_params(params_to_be_logged)
        mlflow.log_artifact(input_path)

        command = [
            "poetry",
            "run",
            "modelgauge",
            "run-csv-items",
            "-s",
            "llama-3-70b-chat",
            input_path,
        ]
        result = subprocess.run(command, capture_output=False, text=True)

        output_file_path = find_output_file_path(input_path)

        # log the output file path as an artifact
        mlflow.log_artifact(output_file_path)

def find_output_file_path(input_path):
        output_file_path = None

        for file in pathlib.Path(os.path.dirname(input_path)).iterdir():
            base_name = os.path.basename(input_path).split(".")[0]

            if re.match(r"{base_name}-\d{8}-\d{6}.csv", file.name):
                output_file_path = file
                break

        return output_file_path





if __name__ == "__main__":
    main()
