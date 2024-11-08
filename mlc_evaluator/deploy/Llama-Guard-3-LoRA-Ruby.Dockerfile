# (from https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)

# Pull the vllm image
FROM vllm/vllm-openai

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

ARG ADAPTER_URL=https://MLCommons-Association:${HF_TOKEN}@huggingface.co/MLCommons-Association/Llama-Guard-3-LoRA-Ruby
ARG ADAPTERS_COMMIT_HASH=3b4af22a6f018252d709809c246e33b6071ce1ec

RUN echo "Building Llama Guard 3 LoRA finetuned vLLM image..."

# Download the LoRA adapter weights from HF
RUN apt-get update && apt-get install -y git-lfs && git lfs install
RUN echo "Cloning adapter from MLCommons-Association huggingface repo"
RUN git clone ${ADAPTER_URL} /root/adapters

# Checkout specific commit
RUN cd /root/adapters && git checkout ${ADAPTERS_COMMIT_HASH}

# Start the vllm server with lora adapters
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Llama-Guard-3-8B" ,"--enable-lora", "--lora-modules", "llama-guard-3-lora=/root/adapters"]
