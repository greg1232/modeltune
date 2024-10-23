# (from https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)

# Pull the vllm image
FROM vllm/vllm-openai

# Copy over the adapters we want
ARG ADAPTER_DIR
RUN echo "Building Llama Guard 3 LoRA finetuned vLLM image..."
RUN test -n "$ADAPTER_DIR"
RUN echo "Copying adapter from: $ADAPTER_DIR"
COPY $ADAPTER_DIR /root/adapters

# Start the vllm server with lora adapters
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Llama-Guard-3-8B" ,"--enable-lora", "--lora-modules", "llama-guard-3-lora=/root/adapters"]
