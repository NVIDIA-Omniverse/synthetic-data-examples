## Getting started

### Install Dependencies

- [`docker-compose`](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Run the labs

```
bash run.sh
```

The script creates a local `.env` file with random Jupyter tokens and prints the access URLs. Keep this file private; delete it and rerun the script to rotate the tokens.

### Access the labs
The labs are available only from the local machine:

- Part 1: Generate synthetic data with Omniverse — `http://127.0.0.1:8882/lab?token=<printed-token>`
- Part 2: Training a model with synthetic data — `http://127.0.0.1:8883/lab?token=<printed-token>`
- Part 3: Deploy model to Triton — `http://127.0.0.1:8884/lab?token=<printed-token>`
