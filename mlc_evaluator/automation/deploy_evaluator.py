# CLI to run an evaluator Docker container on a production machine.
# This mostly wraps the functions in the gcp module, which decouples those functions
# from the user tool calling them (this CLI, but also a REST API in the future, potentially)

import logging
import os

import click

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
    default=gcp.IP_ADDRESS,
    help="Hostname or IP address of instance",
)
def echo(hostname: str = gcp.IP_ADDRESS):
    instance = gcp.Instance(hostname)
    response = gcp.remote_command("echo 1", instance)
    logger.info(str(response))


@click.command()
@click.option("-i", "--image", required=True)
@click.option("-t", "--tag", required=False, default="latest")
@click.option(
    "-h",
    "--hostname",
    default=gcp.IP_ADDRESS,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=gcp.INSTANCE_NAME,
    help="Instance name, if known",
)
def pull(
    image: str,
    tag: str = "latest",
    hostname: str = gcp.IP_ADDRESS,
    name: str = gcp.INSTANCE_NAME,
):
    assert name or hostname
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
    default=gcp.IP_ADDRESS,
    help="Instance hostname or IP address, if known",
)
@click.option(
    "-n",
    "--name",
    default=gcp.INSTANCE_NAME,
    help="Instance name, if known",
)
def run(
    image: str,
    tag: str = "latest",
    hostname: str = gcp.IP_ADDRESS,
    name: str = gcp.INSTANCE_NAME,
):
    assert name or hostname
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
        print()


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
        print()
    except Exception as exc:
        logger.error(f"Unable to find instance named {name}: {exc}.")


cli.add_command(echo)
cli.add_command(instance)
cli.add_command(instances)
cli.add_command(pull)
cli.add_command(run)
cli.add_command(start)
cli.add_command(stop)


if __name__ == "__main__":
    cli()
