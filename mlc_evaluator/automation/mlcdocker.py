import os

# DEFAULT_NAMESPACE = "mlcommons"

DEFAULT_NAMESPACE = "mlcommons"
DEFAULT_REGISTRY = "ghcr.io"
DEFAULT_MODEL = "llamaguard-lora"


def image_uri(
    image: str,
    tag: str = "latest",
    namespace: str = DEFAULT_NAMESPACE,
    registry: str = DEFAULT_REGISTRY,
) -> str:
    return f"{registry}/{namespace}/{image}:{tag}"


def login_cmd(token: str, user: str, registry: str = DEFAULT_REGISTRY) -> str:
    return f"docker login -u {user} -p {token} {registry}"


def pull_cmd(image: str, tag: str) -> str:
    return f"docker pull {image_uri(image=image, tag=tag)}"


def what_is_running_cmd():
    return "docker ps -a"


def run_cmd(image: str, tag: str) -> str:
    api_key = os.getenv("VLLM_API_KEY", "secret_key")
    parts = (
        "docker run",
        "--runtime nvidia",
        "--gpus all",
        # fmt: off
        '--env "HUGGING_FACE_HUB_TOKEN=' + os.getenv("HF_TOKEN") + '"',
        # fmt: on
        "-p 8000:8000",
        "--ipc=host",
        image_uri(image, tag),
        f"--api_key={api_key}",
        "--dtype=half",
    )
    return " ".join(parts)


def base_url(hostname: str):
    return f"http://{hostname}:8000/v1"
