# Interact with GCP
from dataclasses import dataclass
import os
from pathlib import Path

import paramiko

EVALUATOR_RUNNER_TAG = {"mlc-machine-type": "evaluator"}

IP_ADDRESS = "34.45.76.162"
SSH_USER = "admin"
SSH_KEY_FILENAME = str(
    Path(os.getenv("HOME", os.path.expanduser("~")))
    / ".ssh"
    / "eval-runner-01-dev-admin"
)


@dataclass
class Instance:
    hostname: str


def _list_instances(tags: dict = EVALUATOR_RUNNER_TAG):
    # request = service.instances().list(project=project, zone=zone, filter='labels.my-label=my-value')
    pass


def find_instance(instances):
    pass


def create_instance(name: str):
    pass


def configure_instance(instance):
    pass


def install_drivers(instance):
    pass


def set_config(instance, config: dict):
    pass


def ssh_command(
    command: str,
    instance: Instance,
    username: str = SSH_USER,
    key_filename: str = SSH_KEY_FILENAME,
) -> list:
    output = []
    with paramiko.SSHClient() as client:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            instance.hostname,
            username=username,
            key_filename=key_filename,
        )
        _, stdout, _ = client.exec_command(command)
        output = [line.rstrip() for line in stdout]
    return output
