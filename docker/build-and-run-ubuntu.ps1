$NAME = "autoscoper_dev_ubuntu"
$TAG = "latest"
$IMAGENAME = "${NAME}:$TAG"

docker build -t $IMAGENAME -f $PSScriptRoot/UbuntuDockerFile .

docker run --rm -it --gpus all --name "autoscoper_ubuntu" "autoscoper_dev_ubuntu:latest"