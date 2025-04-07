#!/bin/bash

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 image_name container_name"
  exit 1
fi

UID=`id -u`
GID=`id -g`
gpu_id=5
USER_NAME=${USER}
IMG_NAME=$1
MY_IMG_NAME=${IMG_NAME}_${USER_NAME}
CONTAINER_NAME=$2

echo "uid = $UID"
echo "gid = $GID"
echo "user: $USER_NAME"
echo "docker name: $DOCKER_NAME"
echo "my docker name: $MY_DOCKER_NAME"

# customerize the docker image for specific folder mount
# use my own username instead of root
docker build --build-arg uid=$UID --build-arg gid=$GID --build-arg USER_NAME=$USER_NAME --build-arg \
INPUT_DOCKER=$IMG_NAME --build-arg DEVICE_ID=$gpu_id -t $MY_IMG_NAME -f Dockerfile.wrapper .

# create a docker container using the customerized docker image
docker run -it --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
--shm-size=64G --device=/dev/kfd --device=/dev/dri  -v  $HOME/Workplace:/workspace --name $CONTAINER_NAME $MY_IMG_NAME
