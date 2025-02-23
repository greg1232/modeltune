# CLI to run an evaluator Docker container on a production machine.
# This mostly wraps the functions in the gcp module, which decouples those functions
# from the user tool calling them (this CLI, but also a REST API in the future, potentially)

import logging
import os

import click
import gcp
import mlcdocker
from openai import OpenAI

logger = logging.getLogger("evaluator-deploy")
logging.basicConfig(level=logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
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
    """Configures environment variables on a remote instance. Only needed once."""
    gcp.configure_environment(hostname, name, force)


@cli.command()
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
    """Runs a simple echo command on a remote instance for testing."""
    hostname = gcp.find_ip_address(hostname, name)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command("echo 1", instance)
    logger.info(str(response))


@cli.command()
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
    """Pulls the specified image and tag on the remote instance."""
    hostname = gcp.find_ip_address(hostname, name)
    login_cmd = mlcdocker.login_cmd(
        token=os.getenv("CR_PAT"), user=os.getenv("CR_USER")
    )
    pull_cmd = mlcdocker.pull_cmd(image, tag)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command(f"{login_cmd};{pull_cmd}", instance)
    logger.info(str(response))


@cli.command()
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
    """Runs the specified image on the remote instance."""
    hostname = gcp.find_ip_address(hostname, name)
    run_cmd = mlcdocker.run_cmd(image, tag)
    logger.debug(run_cmd)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command(run_cmd, instance)
    logger.info(str(response))


@cli.command()
@click.option("-n", "--name", required=True, help="Instance name")
def start(name: str):
    """Starts the cloud instance, if found and not already started."""
    gcp.start_instance(name)


@cli.command()
@click.option("-n", "--name", required=True, help="Instance name")
def stop(name: str):
    """Stops the cloud instance, if found and not already stopped."""
    gcp.stop_instance(name)


@cli.command()
def instances():
    """Displays a list of all the instances in the default zone."""
    instances = gcp.list_instances()
    for instance in instances:
        print(f"Name: {instance['name']}")
        print(f"ID: {instance['id']}")
        print(f"IP Address: {instance['ip_address']}")
        print(f"Status: {instance['status']}")


@cli.command()
@click.option("-n", "--name", required=True, help="Instance name")
def instance(name: str):
    """Displays basic information about the specified instance."""
    instances = gcp.list_instances()
    try:
        # TODO: pull the IP from the Instance object we already have
        # rather than look it up
        the_instance = [i for i in instances if i["name"] == name][0]
        ip_address = gcp.get_ip_from_instance_name(the_instance["name"])
        print(f"Name: {the_instance['name']}")
        print(f"ID: {the_instance['id']}")
        print(f"IP Address: {ip_address}")
        print(f"Status: {the_instance['status']}")
    except Exception as exc:
        logger.error(f"Unable to find instance named {name}: {exc}.")


@cli.command()
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
@click.option(
    "-m",
    "--model",
    default=mlcdocker.DEFAULT_MODEL,
    help="Model name",
)
def test(hostname: str, name: str, prompt: str, model: str = mlcdocker.DEFAULT_MODEL):
    """Sends a test prompt to the vllm server."""
    hostname = gcp.find_ip_address(hostname, name)
    base_url = mlcdocker.base_url(hostname)
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("VLLM_API_KEY", "fake key"),
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        print(completion.choices[0].message)
    except Exception as exc:
        logger.error(f"Error sending prompt: {exc}")


@cli.command()
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
def what_is_running(hostname: str, name: str):
    """What Docker is running on the remote instance, if anything."""
    hostname = gcp.find_ip_address(hostname, name)
    cmd = mlcdocker.what_is_running_cmd()
    logger.debug(cmd)
    instance = gcp.Instance(hostname)
    response = gcp.remote_command(cmd, instance)
    logger.info(str(response))


@cli.command()
@click.option(
    "-n",
    "--name",
    default=None,
    help="Returns a URL to the specified instance, or first running instance we found",
)
def url(name: str):
    """This is to enable commands like curl $(python deploy_evaluator.py url)/..."""
    url = mlcdocker.base_url("localhost")

    if name:
        try:
            hostname = gcp.find_ip_address(name=name)
            if hostname:
                url = mlcdocker.base_url(hostname)
        except:
            pass
    else:
        instances = gcp.list_instances(running_only=True, max_instances=1)
        if instances:
            instance = instances[0]
            hostname = instance["ip_address"]
            url = mlcdocker.base_url(hostname)
    print(url)  # we always want to print something regardless of log level
    return url


if __name__ == "__main__":
    cli()
