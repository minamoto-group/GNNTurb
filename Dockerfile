FROM asha225/libtorch-pyg:11.6.2-cudnn8-devel-ubuntu20.04

# Install OpenFOAM v2112
RUN wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash && \
    apt-get -y install openfoam2112-default

RUN mkdir /work
WORKDIR /work

ARG GROUPID
ARG USERID
ARG USERNAME

# Add user
RUN useradd -m -s /bin/bash -u ${USERID} -g ${GROUPID} ${USERNAME} \
    && chown -R ${USERID}:${USERID} /work
USER ${USERNAME}

# Add OpenFOAM's configuration
RUN sed -i '$a source /usr/lib/openfoam/openfoam2112/etc/bashrc' ~/.bashrc