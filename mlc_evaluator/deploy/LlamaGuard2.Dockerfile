# (from https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html)
# TODO Dockerfile currently only supports LlamaGuard2 as the base model. Create different Dockerfiles for different base models in the future

# Pull the vllm image
FROM vllm/vllm-openai

# Copy over the adapters we want
ARG ADAPTER_DIR
RUN test -n "$ADAPTER_DIR"
RUN echo "Copying adapter from: $ADAPTER_DIR"
COPY $ADAPTER_DIR /root/adapters

# Start the vllm server with lora adapters
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "meta-llama/Meta-Llama-Guard-2-8B" ,"--enable-lora", "--lora-modules", "llamaguard-lora=/root/adapters"]
