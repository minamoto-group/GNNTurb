# GNNTurb: GNN-based SGS stress model in OpenFOAM

## Docker

GNNTurb can be compiled and run in a docker image with [OpenFOAM v2112](https://www.openfoam.com/) and CUDA. 
This docker image is based on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) (11.6.2-cudnn8-devel-ubuntu20.04) and includes the following libraries:
- [LibTorch](https://github.com/pytorch/pytorch) 1.12.1
- [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) 2.1.0
- [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.14
- [pytorch_spline_conv](https://github.com/rusty1s/pytorch_spline_conv) 1.2.1

Example of creating a docker container:
1. `$ docker build -t <custom image name> --build-arg USERID="$(id -u $USER)" --build-arg GROUPID="$(id -g $USER)" --build-arg USERNAME="$(id -u $USER --name)" .`
2. `$ docker run --rm -it --init --runtime=nvidia -v ${PWD}:/work <custom image name> /bin/bash`


## Reference

Asahi Abekawa , Yuki Minamoto , Kosuke Osawa , Haruya Shimamoto , and Mamoru Tanahashi , "Exploration of robust machine learning strategy for subgrid scale stress modelling", Physics of Fluids (in press) (2023); https://doi.org/10.1063/5.0134471
