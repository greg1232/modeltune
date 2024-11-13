# (from https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)

# Pull the vllm image
FROM vllm/vllm-openai

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

ARG ADAPTER_URL=https://MLCommons-Association:${HF_TOKEN}@huggingface.co/MLCommons-Association/Mistral-7B-Ruby
ARG ADAPTERS_COMMIT_HASH=f7fd29bf1d1e99d8de36a768559931e5d0e846ac

RUN echo "Building Mistral LoRA finetuned vLLM image..."

# Download the LoRA adapter weights from HF
RUN apt-get update && apt-get install -y git-lfs && git lfs install
RUN echo "Cloning adapter from MLCommons-Association huggingface repo"
RUN git clone ${ADAPTER_URL} /root/adapters

# Checkout specific commit
RUN cd /root/adapters && git checkout ${ADAPTERS_COMMIT_HASH}

# Start the vllm server with lora adapters
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "mistralai/Mistral-7B-Instruct-v0.3" ,"--enable-lora", "--lora-modules", "mistral7b-lora=/root/adapters"]
