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


## A Posteriori Testing

The considered flow condition and configuration are based on a well-known DNS of the $Re_{\tau} = 395$ turbulent channel flow ([Moser et al.](https://aip.scitation.org/doi/10.1063/1.869966)).
The simulation domain is cuboid with periodic boundaries in both streamwise and spanwise directions and no-slip wall boundary conditions in the transverse direction.
The domain dimensions are $2 \pi \delta_{chw} \times 2 \delta_{chw} \times \pi \delta_{chw}$ $(\delta_{chw} = 1.0 (\mathrm{m}))$, which are discretized onto $65 \times 48 \times 65$ grid points.
The wall distance to the first grid point in the wall direction $\min(\Delta y^+) = 1.31$ and $\max(\Delta y^+) = 60.0$.
In the other directions, $\Delta x^+ = 37.9$ and $\Delta z^+ = 19.0$.

A LES using the trained gnn model $\texttt{GNN-ALL}$ (denoted as LES G1) was performed.
Additionally, the computational domain was rotated $\theta_z = \pi/6$ around the $z$-axis, and LES using $\texttt{GNN-ALL}$ (denoted as LES G1') was also performed.

- Velocity distributions and isosurfaces of the second invariant of the velocity gradient tensor of the LES G1' result:

<p align="center">
  <img width="70%" src="http://drive.google.com/uc?export=view&id=18JHLCFCZdNFH976UmvoA98O7rrkDxn1a" />
</p>

- Comparison of the DNS data and LES solutions in terms of (a) the mean streamwise velocity profile, and the (b) total, (c) GS, and (d) modeled SGS Reynolds shear stresses:

<p align="center">
  <img width="100%" src="http://drive.google.com/uc?export=view&id=1WW-6yUyeVfEHVRNZza2A0BajK3DBQaaH" />
</p>


## Reference

A. Abekawa, Y. Minamoto, K. Osawa, H. Shimamoto, and M. Tanahashi , "Exploration of robust machine learning strategy for subgrid scale stress modeling", Physics of Fluids 35, 015162 (2023) https://doi.org/10.1063/5.0134471
