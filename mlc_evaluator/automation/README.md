# How to Deploy and Run Evaluators to GCP

```bash
poetry install
poetry run python deploy_evaluator.py --help # this will show help
```

## Prerequisites

* gcloud CLI
* gcloud CLI active login `gcloud auth application-default login --no-launch-browser`
* a VM at GCP (details at bottom)
* private ssh key to a machine in `~/.ssh/eval-runner-dev-admin` (`chmod 0600` as usual).

### Environment Variables On *Your Machine*:


```bash
# optional (sensible defaults in gcp.py)
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1a"
# recommended (sensible defaults in gcp.py)
export GCP_PROJECT="ai-safety-dev"
export SSH_USER="admin"
# github personal access token with read:packages permission
export CR_PAT=<your github token>
export CR_USER=<your github username>
# hugging face token
export HF_TOKEN=<your token>
export HF_USER=MLCommons-Association
export VLLM_API_KEY=<the vllm server api key> # in Keeper
```

### Environment Variables On *The Server*:

These can be set from your computer by calling `poetry run python deploy_evaluator.py configure -h <instance ip>`.
You only need to do it once (or if you need to set new values).

For reference:

```bash
# github personal access token with read:packages permission
export CR_PAT=<your github token>
export CR_USER=<your github username>
# hugging face token
export HF_TOKEN=<your token>
export HF_USER=MLCommons-Association
export VLLM_API_KEY=<the vllm server api key> # in Keeper
```

## Typical Scenarios

### What Images Can I Use?

[Visit the registry on Github](https://github.com/orgs/mlcommons/packages?tab=packages&q=ws3)

In these examples, your image is `ws3-llama-guard-3-ruby` and the tag is `v0.3`.

### Pull An Image

```
poetry run python deploy_evaluator.py pull -i ws3-llama-guard-3-ruby -t v0.3 -h <instance ip>
```

### Run A VLLM Container

```
poetry run python deploy_evaluator.py run -i ws3-llama-guard-3-ruby -t v0.3 -h <instance ip>
```

### What Instances Do I Have?

```bash
python deploy_evaluator.py instances
```

### What Is My Instance's IP Address?

```bash
python deploy_evaluator.py instance -n eval-runner-03-dev-vm
```

### Turn My Instance On

```bash
python deploy_evaluator.py start -n eval-runner-03-dev-vm
```

### Turn My Instance Off

```bash
python deploy_evaluator.py stop -n eval-runner-03-dev-vm
```

<a name="models"></a>### Get The List Of Models

```bash
curl $(poetry run python deploy_evaluator.py url)/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY"
```

### More Commands

```bash
poetry run python deploy_evaluator.py --help
poetry run python deploy_evaluator.py <command> --help
```

## Troubleshooting

If something isn't working, the first thing to check are permissions.

### Permissions

* On your computer, try `ssh admin@<instance ip> -i ~/.ssh/eval-runner-dev-admin`. If that doesn't work, make sure you are using
  the right private key, the private key file is `chmod 0600`, and the instance IP address is correct. The private
  key is in Keeper under `eval-runner-01-dev-shared`
* On the machine, run `docker login -u <your github username> -p ${CR_PAT} ghcr.io`. If this fails, your username
  or token are not correct.
* On the machine, after successful `docker login`, try pulling: `docker pull ghcr.io/mlcommons/ws3-mistral-lora-ruby:v0.1`.
  If that doesn't work, your github token is missing the `read:packages` permission, or it is expired.
* On your computer, send a test prompt. If you get an error, run all the tests under "Troubleshooting." If that still
  doesn't work, contact AIRR engineering.

#### Send A Test Prompt Via CURL

If you don't know what models are available, [look them up](#models).

```bash
curl $(poetry run python deploy_evaluator.py url)/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{
  "model": "llama-guard-3-lora",
  "prompt": "Pangolins are",
  "max_tokens": 7,
  "temperature": 0
  }'
```

#### Send A Test Prompt Via The Deployer

```bash
# run a test prompt
poetry run python deploy_evaluator.py test -h 34.133.88.36 -p "what time is it?" -m mistral-lora-ruby`
```

### Configuration

* If the instance is brand new, and the `run` or `pull` commands fail, try running `python deploy_evaluator.py configure -n <instance_name>`
* The `run` command won't show anything until the container is stopped. It's recommended to background it, and check that the container is indeed running with the `what-is-running` command.

```bash
python deploy_evaluator.py run -i ws3-model-test -t latest -n eval-runner-03-dev-vm &
python deploy_evaluator.py what-is-running -n eval-runner-03-dev-vm
python deploy_evaluator.py test -n eval-runner-03-dev-vm -p "I will smoke crystal meth"

curl $(poetry run python deploy_evaluator.py url)/models \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY"
```



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

## TODO Next

* Stream logs for long-running operations like `pull` or `run` so the operator can see what's happening.
* `async` the long-running operations.
* Better format for `what-is-running` command output.
* Figure out how to:
  1. send the system prompt in the `test` request (should it be under "system" role?)
  2. verify that the right system prompt is sent to the right version of evaluator requested.
