# CLI to run an evaluator Docker container on a production machine.
# This mostly wraps the functions in the gcp module, which decouples those functions
# from the user tool calling them (this CLI, but also a REST API in the future, potentially)

import logging
import os

import click
from openai import OpenAI

import gcp
import mlcdocker

logger = logging.getLogger("evaluator-deploy")
logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "-h",
    "--hostname",
    default=None,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Instance name, if known",
)
@click.option("-f", "--force", is_flag=True, default=False)
def configure(hostname: str = None, name: str = None, force: bool = False) -> None:
    """Normally you wouldn't need to use this once the machine has been configured.
    But if you're running into issues with missing config variables, use this."""
    gcp.configure_environment(hostname, name, force)


@click.command()
@click.option(
    "-h",
    "--hostname",
    default=None,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Instance name, if known",
)
def echo(hostname: str = None, name: str = None):
    hostname = gcp.find_ip_address(hostname, name)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command("echo 1", instance)
    logger.info(str(response))


@click.command()
@click.option("-i", "--image", required=True)
@click.option("-t", "--tag", required=False, default="latest")
@click.option(
    "-h",
    "--hostname",
    default=None,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Instance name, if known",
)
def pull(
    image: str,
    tag: str = "latest",
    hostname: str = None,
    name: str = None,
):
    hostname = gcp.find_ip_address(hostname, name)
    login_cmd = mlcdocker.login_cmd(
        token=os.getenv("CR_PAT"), user=os.getenv("CR_USER")
    )
    pull_cmd = mlcdocker.pull_cmd(image, tag)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command(f"{login_cmd};{pull_cmd}", instance)
    logger.info(str(response))


@click.command()
@click.option("-i", "--image", required=True)
@click.option("-t", "--tag", required=False, default="latest")
@click.option(
    "-h",
    "--hostname",
    default=None,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Instance name, if known",
)
def run(
    image: str,
    tag: str = "latest",
    hostname: str = None,
    name: str = None,
):
    hostname = gcp.find_ip_address(hostname, name)
    run_cmd = mlcdocker.run_cmd(image, tag)
    logger.info(run_cmd)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command(run_cmd, instance)
    logger.info(str(response))


@click.command()
@click.option("-n", "--name", required=True, help="Instance name")
def start(name: str):
    """Starts the cloud instance, if found and not already started."""
    gcp.start_instance(name)


@click.command()
@click.option("-n", "--name", required=True, help="Instance name")
def stop(name: str):
    """Stops the cloud instance, if found and not already stopped."""
    gcp.stop_instance(name)


@click.command()
def instances():
    instances = gcp.list_instances()
    for instance in instances:
        print(f"Name: {instance['name']}")
        print(f"ID: {instance['id']}")
        print(f"Status: {instance['status']}")


@click.command()
@click.option("-n", "--name", required=True, help="Instance name")
def instance(name: str):
    instances = gcp.list_instances()
    try:
        # TODO: pull the IP from the Instance object we already have
        # rather than look it up
        the_instance = [i for i in instances if i["name"] == name][0]
        ip = gcp.get_ip_from_instance_name(the_instance["name"])
        print(f"Name: {the_instance['name']}")
        print(f"IP: {ip}")
        print(f"ID: {the_instance['id']}")
        print(f"Status: {the_instance['status']}")
    except Exception as exc:
        logger.error(f"Unable to find instance named {name}: {exc}.")


@click.command()
@click.option(
    "-h",
    "--hostname",
    default=None,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Instance name, if known",
)
@click.option(
    "-p",
    "--prompt",
    required=True,
    help="Prompt",
)
def test(hostname: str, name: str, prompt: str):
    hostname = gcp.find_ip_address(hostname, name)
    base_url = f"http://{hostname}:8000/v1"
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("VLLM_API_KEY", "fake key"),
        )
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-v0.1",
            messages=[{"role": "user", "content": prompt}],
        )
        print(completion.choices[0].message)
    except Exception as exc:
        logger.error(f"Error sending prompt: {exc}")


cli.add_command(configure)
cli.add_command(echo)
cli.add_command(instance)
cli.add_command(instances)
cli.add_command(pull)
cli.add_command(run)
cli.add_command(start)
cli.add_command(stop)
cli.add_command(test)

if __name__ == "__main__":
    cli()
