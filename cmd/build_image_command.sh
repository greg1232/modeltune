docker_build_command="docker build -t modeltune:latest ."

# Run docker build command
echo $(green_bold Building image with command: ${docker_build_command})
eval $docker_build_command

echo $(green_bold Successfully built image)


