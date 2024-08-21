import os

# DEFAULT_NAMESPACE = "mlcommons"

DEFAULT_NAMESPACE = "mlcommons"
DEFAULT_REGISTRY = "ghcr.io"


def image_uri(
    image: str,
    tag: str = "latest",
    namespace: str = DEFAULT_NAMESPACE,
    registry: str = DEFAULT_REGISTRY,
) -> str:
    return f"{registry}/{namespace}/{image}:{tag}"


def login_cmd(token: str, user: str, registry: str = DEFAULT_REGISTRY) -> str:
    return f"docker login -u {user} -p {token} {DEFAULT_REGISTRY}"


def pull_cmd(image: str, tag: str) -> str:
    return f"docker pull {image_uri(image=image, tag=tag)}"


def what_is_running_cmd():
    return "docker ps -a"


def run_cmd(image: str, tag: str) -> str:
    # Ryan's vllm
    # parts = (
    #     "docker run",
    #     "--runtime nvidia",
    #     "--gpus all",
    #     # fmt: off
    #     '--env "HUGGING_FACE_HUB_TOKEN=' + os.getenv("HF_TOKEN") + '"',
    #     # fmt: on
    #     "-p 8000:8000",
    #     "--ipc=host",
    #     image_uri(image, tag),
    #     f"--api_key={os.getenv("VLLM_API_KEY", "secret_key")}",
    #     "--dtype=half"
    # )

    # default vllm
    parts = (
        "source /home/admin/.bashrc &&",
        "docker run --runtime nvidia --gpus all",
        "-v ~/.cache/huggingface:/root/.cache/huggingface",
        # fmt: off
        f'--env "HUGGING_FACE_HUB_TOKEN={os.getenv("HF_TOKEN", "")}"',
        "-p 8000:8000",
        "--ipc=host",
        "vllm/vllm-openai:latest",
        "--model mistralai/Mistral-7B-v0.1",
        "--dtype=half"
    )
    return " ".join(parts)
