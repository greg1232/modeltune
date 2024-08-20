# How to Deploy and Run Evaluators to GCP

```bash
poetry install
python deploy_evaluator.py # this will show help
```

## Prerequisites

* gcloud CLI
* gcloud CLI active login `gcloud auth application-default login --no-launch-browser`
* private ssh key to a machine in `~/.ssh/eval-runner-dev-admin`
* Environment variables on *your machine*:


```bash
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1a"
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


## Example Usage

### Run the "evaluator" image on the "eval-runner-01-dev-vm" machine

```bash
python deploy_evaluator.py run -n eval-runner-01-dev-vm -i evaluator
```

### What instances do I have?

```bash
python deploy_evaluator.py instances
```

### What is the IP address of my instance?

```bash
python deploy_evaluator.py instance -n eval-runner-01-dev-vm
```

### Turn my instance on

```bash
python deploy_evaluator.py start -n eval-runner-01-dev-vm
```


### Turn my instance off

```bash
python deploy_evaluator.py stop -n eval-runner-01-dev-vm
```

## How It Works

The [deploy_evaluator.py](./deploy_evaluator.py) program issues commands via the GCP
SDK (e.g. when listing instances) or via ssh to the instance itself, and displays
stdout from the instance in that case.

## Network and Other Bits

* The instance needs the `vllm-8000` network tag to serve requests over :8000.
