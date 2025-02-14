
./modeltune build-image

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Get the project root directory
ROOT_DIRECTORY=$LOCAL_DIRECTORY/..

# Make the mlruns directory
mkdir -p $ROOT_DIRECTORY/experiments

declare -a sut_inference_command_parts

sut_inference_command_parts=(
    "cd /app/annotations/auto_annotation" "&&" "poetry" "run" "python3" "/app/annotations/auto_annotation/mlflow/getting_responses_from_SUTs.py" ";"
)

declare -a docker_command_parts
docker_command_parts=("docker" "run" "-v" "$ROOT_DIRECTORY/experiments:/app/experiments"
    "--rm" "-e" "TOGETHER_API_KEY=xyz"
    "-e" "MLFLOW_TRACKING_URI=file:///app/experiments/mlruns" )

docker_command_parts+=("modeltune:latest" "sh" "-c" "'${sut_inference_command_parts[*]}'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command
