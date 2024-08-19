# Interact with GCP
# Some of these functions are convenience wrappers around the lower-level functions
# provided by the GCP SDK. They handle the project and zone so that your client
# app doesn't have to bother with that.

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import paramiko

from vendor import google

# TODO: use this tag on evaluator machines to look them up by tag
# This may be useful if we ever have more than one machine
EVALUATOR_RUNNER_TAG = {"mlc-machine-type": "evaluator"}

IP_ADDRESS = "35.226.32.27"
INSTANCE_NAME = "eval-runner-01-dev-vm"
SSH_USER = "admin"
SSH_KEY_FILENAME = str(
    Path(os.getenv("HOME", os.path.expanduser("~")))
    / ".ssh"
    / "eval-runner-01-dev-admin"
)

logger = logging.getLogger("gcp")
logging.basicConfig(level=logging.INFO)


@dataclass
class Instance:
    hostname: str


def configure_instance(instance):
    pass


def install_drivers(instance):
    pass


def set_config(instance, config: dict):
    pass


def list_instances():
    project = os.getenv("GCP_PROJECT", None)
    zone = os.getenv("GCP_ZONE", None)
    assert project and zone
    instances = google.list_all_instances(project)

    instance_summary = []
    try:
        zone_instances = instances.get(f"zones/{zone}", [])
        for instance in zone_instances:
            instance_summary.append(
                {"id": instance.id, "name": instance.name, "status": instance.status}
            )
    except Exception as exc:
        logger.error(f"Unable to look up instances in zone {zone}: {exc}.")
    return instance_summary


def get_ip_from_instance_name(name: str) -> str:
    project = os.getenv("GCP_PROJECT", None)
    zone = os.getenv("GCP_ZONE", None)
    assert project and zone

    ip = ""
    try:
        ins = google.get_instance(project, zone, name)
        ip = google.get_instance_ip_address(ins, google.IPType.EXTERNAL).pop()
    except Exception as exc:
        logger.error(f"Unable to look up IP address for instance named {name}: {exc}.")
        raise
    return ip


def start_instance(name: str) -> None:
    project = os.getenv("GCP_PROJECT", None)
    zone = os.getenv("GCP_ZONE", None)
    assert project and zone

    try:
        ins = google.get_instance(project, zone, name)
        if ins.status == "STOPPED":
            google.start_instance(project, zone, name)
    except Exception as exc:
        logger.error(f"Unable to start instance named {name}: {exc}.")
        raise


def stop_instance(name: str) -> None:
    project = os.getenv("GCP_PROJECT", None)
    zone = os.getenv("GCP_ZONE", None)
    assert project and zone

    try:
        ins = google.get_instance(project, zone, name)
        if ins.status == "RUNNING":
            google.stop_instance(project, zone, name)
    except Exception as exc:
        logger.error(f"Unable to start instance named {name}: {exc}.")
        raise


def remote_command(
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
