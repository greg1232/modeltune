import logging
import os

import click

import mlcdocker
import gcp

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
    response = gcp.ssh_command("echo 1", instance)
    logger.info(str(response))


@click.command()
@click.option("-i", "--image", required=True)
@click.option("-t", "--tag", required=False, default="latest")
@click.option(
    "-h",
    "--hostname",
    default=gcp.IP_ADDRESS,
    help="Hostname or IP address of instance",
)
def pull(image: str, tag: str = "latest", hostname: str = gcp.IP_ADDRESS):
    login_cmd = mlcdocker.login_cmd(
        token=os.getenv("CR_PAT"), user=os.getenv("CR_USER")
    )
    pull_cmd = mlcdocker.pull_cmd(image, tag)
    instance = gcp.Instance(hostname)
    response = gcp.ssh_command(f"{login_cmd};{pull_cmd}", instance)
    logger.info(str(response))


@click.command()
@click.option("-i", "--image", required=True)
@click.option("-t", "--tag", required=False, default="latest")
@click.option(
    "-h",
    "--hostname",
    default=gcp.IP_ADDRESS,
    help="Hostname or IP address of instance",
)
def run(image: str, tag: str = "latest", hostname: str = gcp.IP_ADDRESS):
    run_cmd = mlcdocker.run_cmd(image, tag)
    instance = gcp.Instance(hostname)
    response = gcp.ssh_command(run_cmd, instance)
    logger.info(str(response))


cli.add_command(run)
cli.add_command(pull)
cli.add_command(echo)


if __name__ == "__main__":
    cli()
