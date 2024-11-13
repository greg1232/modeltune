# Deploying custom model

## Overview
This guide will help you prepare an image that can serve our MLC finetuned
models for inference. We use vLLM to handle inference and start a server

TLDR: What's happening at a high level?
1. Download the adapter weights (not necessarily from HuggingFace)
2. Use docker to create a vLLM image with those adapter weights
3. Run that docker image on GCP or AWS to serve the custom model

Requirements
- LoRA weights you want to deploy
- Access to the MLC package (container) registry
- HuggingFace access token with permissions to base model eg Llama Guard 2

Supported
- LoRA finetuned Llama Guard 2

Not supported (yet)
- Mistral models
- Llama Guard 3

## Step 1: Building the image
TODO where should you be building the image? Should be buildable on any environment (unless nvidia gpu is a requirement... why?)
TODO instructions on token management: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
TODO instructions on pushing image with token: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry


### Building a new image and pushing to MLC registry
1. Download LoRA adapter weights and copy them into in the `./adapters`
   directory in the same directory as the Dockerfile (at minimum, need the
   `adapter_config.json` and `adapter_model.safetensors` files. (maybe TODO...
   add details how to get that)
1. Build the docker image, ensure the tag points at the MLC github package registry.

Initially the `<IMAGE_NAME>` is `ws3-model-test`.
```
docker build -t ghcr.io/mlcommons/<IMAGE_NAME> --build-arg HF_TOKEN=<HF_TOKEN> .
```
1. Push the docker image to the MLC package registry
```
docker push ghcr.io/mlcommons/<IMAGE_NAME>:latest
```

## Step 2: Pulling and running the image
### Where to run?
- on a GCP machine with sufficient NVIDIA GPU

### Pulling the image
- Use instructions to login with personal access token https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
```
docker pull ghcr.io/mlcommons/<YOUR_IMAGE>:latest
```


### Running the container image **(WORK IN PROGRESS)**
- Original command adapted from: https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html
- Need to pass some args to the image when starting
- Which HF_TOKEN (HuggingFace token) do I use? Use the one that has access to the base model. (We
  aren't pulling the LoRA adapters from a repo, we're manually uploading them,
  so no need to have access to that one)
- `-p`: Probably fine, but need to figure out exposing container host ports so
  the API is accessible from externally running modelgauge instances
- `--ipc=host`. Recommended by vLLM but not sure why/if it's required
```
docker run \
    --runtime nvidia \
    --gpus all \
    --env "HUGGING_FACE_HUB_TOKEN=<ADD_HF_TOKEN>" \
    -p 8000:8000 \
    --ipc=host \
    ghcr.io/mlcommons/<IMAGE_NAME>:latest \
    --api-key="<YOUR_SUPER_SECURE_API_KEY>"
```
- NOTE: issues with max sequence len, have to specify that argument manually
  with `--max-model-len` to set the context window size. Sometimes the GPU
  cannot support the default, large window as in Llama Guard 3
```
docker run \
    --runtime nvidia \
    --gpus all \
    --env "HUGGING_FACE_HUB_TOKEN=<ADD_HF_TOKEN>" \
    -p 8000:8000 \
    --ipc=host \
    ghcr.io/mlcommons/<IMAGE_NAME>:latest \
    --api-key="<YOUR_SUPER_SECURE_API_KEY>" \
    --max-model-len=<YOUR_LENGTH> (e.g. 18000)
```

### Check if model successfully deployed the finetuned model
source: https://docs.vllm.ai/en/latest/models/lora.html#serving-lora-adapters
vLLM will host both the base model, and the finetuned one side by side.
Note, when inferencing, you must target the finetuned model explicitly using the model name parameter

**Check deployed models**
You should see both the base model and the lora adapter model named
`llamaguard-lora` or similar alongside the base model which in this case would be
meta's llama guard 2
```
curl localhost:8000/v1/models | jq .
```

- NOTE: if unauthorized, add api key to curl header
```
curl -H "Authorization: Bearer <YOUR_API_KEY>" localhost:8000/v1/models | jq .
```

## FAQs
(coming soon?)

## TODO

* Ws3 should finalized the naming and versioning of the evaluator models. Let us have FIXME to decide that should the default model name be.
