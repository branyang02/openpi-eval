# Docker setup

Docker support is provided for hosting the policy server. The simulator clients
run from their environment-specific virtual environments as described in the
top-level README and simulator READMEs.

- Basic Docker installation instructions are [here](https://docs.docker.com/engine/install/).
- Docker must be installed in [rootless mode](https://docs.docker.com/engine/security/rootless/).
- To use your GPU you must also install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The version of docker installed with `snap` is incompatible with the NVIDIA container toolkit, preventing it from accessing `libnvidia-ml.so` ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)). The snap version can be uninstalled with `sudo snap remove docker`.
- Docker Desktop is also incompatible with the NVIDIA runtime ([issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)). Docker Desktop can be uninstalled with `sudo apt remove docker-desktop`.


If starting from scratch and your host machine is Ubuntu 22.04, the convenience
scripts `scripts/docker/install_docker_ubuntu22.sh` and
`scripts/docker/install_nvidia_container_toolkit.sh` install the required Docker
and NVIDIA runtime pieces.

Build the policy-server image and start the container:

```bash
SERVER_ARGS="policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/openpi-libero-9000" \
    docker compose -f scripts/docker/compose.yml up --build
```

To serve with the PyTorch backend:

```bash
SERVER_ARGS="--pytorch policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/openpi-libero-9000" \
    docker compose -f scripts/docker/compose.yml up --build
```

The container mounts the repository at `/app` and
`${OPENPI_DATA_HOME:-~/.cache/openpi}` at `/openpi_assets`.
