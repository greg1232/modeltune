# How to Deploy and Run Evaluators to GCP

```bash
poetry install
python deploy_evaluator.py # this will show help
```

## Prerequisites

* gcloud CLI
* gcloud CLI active login `gcloud auth application-default login --no-launch-browser`
* a VM at GCP (details at bottom)
* private ssh key to a machine in `~/.ssh/eval-runner-dev-admin`
* Environment variables on *your machine*:


```bash
# optional (sensible defaults in gcp.py)
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1a"
# recommended (sensible defaults in gcp.py)
export GCP_PROJECT="ai-safety-dev"
export SSH_USER="admin"
# github container registry token
export CR_PAT=<your token>
export CR_USER=<your username>
# hugging face token
export HF_TOKEN=<your token>
export HF_USER=MLCommons-Association
export VLLM_API_KEY=<the vllm server api key> # in Keeper
```

* Environment variables on *the server*:

```bash
# github container registry token
export CR_PAT=<your token>
export CR_USER=<your username>
# hugging face token
export HF_TOKEN=<your token>
export HF_USER=MLCommons-Association
export VLLM_API_KEY=<the vllm server api key> # in Keeper
```

## Tests

Not a lot for now.

```bash
python -m pytest tests
```

## Example Usage

### Run the "evaluator" image on the "eval-runner-03-dev-vm" machine

```bash
python deploy_evaluator.py run -n eval-runner-03-dev-vm -i evaluator
```

### What instances do I have?

```bash
python deploy_evaluator.py instances
```

### What is the IP address of my instance?

```bash
python deploy_evaluator.py instance -n eval-runner-03-dev-vm
```

### Turn my instance on

```bash
python deploy_evaluator.py start -n eval-runner-03-dev-vm
```

### Turn my instance off

```bash
python deploy_evaluator.py stop -n eval-runner-03-dev-vm
```

There are more functions too.

## How It Works

The [deploy_evaluator.py](./deploy_evaluator.py) program issues commands via the GCP
SDK (e.g. when listing instances) or via ssh to the instance itself, and displays
stdout from the instance in that case.

## VM, Network and Other Bits

* You need to provision an instance yourself and keep track of the zone.

  * Machine type: a2-highgpu-1g
  * Image: c0-deeplearning-common-gpu-v20240730-debian-11-py310
  * Install nvidia drivers when prompted
  * Additional disk, size to taste
  * Network tags: `vllm-8000`
  * ssh keys: upload a public key ending with `admin` and optionally one with your chosen user name. This will create the user accounts on the machine when it spins up.
    * You will need the matching private key at `~/.ssh/eval-runner-dev-admin` on your machine
      * You can change this in the `gcp.py` module if necessary

* The vllm server will be listening on port 8000

## Troubleshooting

* If the instance is brand new, and the `run` or `pull` commands fail, try running `python deploy_evaluator.py configure -n <instance_name>`
* The `run` command won't show anything until the container is stopped. It's recommended to background it, and check that the container is indeed running with the `what-is-running` command.

```bash
python deploy_evaluator.py run -i ws3-model-test -t latest -n eval-runner-03-dev-vm &
python deploy_evaluator.py what-is-running -n eval-runner-03-dev-vm
python deploy_evaluator.py test -n eval-runner-03-dev-vm -p "I will smoke crystal meth"
```

## TODO Next

* Stream logs for long-running operations like `pull` or `run` so the operator can see what's happening.
* `async` the long-running operations.
* Better format for `what-is-running` command output.
* Figure out how to 1. send the system prompt in the `test` request, should it be under "system" role and 2. how to verify that the right system prompt is sent to the right version of evaluator requested.
*