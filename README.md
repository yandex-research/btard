# Secure Distributed Training at Scale

This repository contains the implementation of experiments from the paper

**"Secure Distributed Training at Scale"**

_Eduard Gorbunov*, Alexander Borzunov*, Michael Diskin, Max Ryabinin_

[**[PDF]** arxiv.org](https://arxiv.org/pdf/2106.11257)

## Overview

The code is organized as follows:

- __[`./resnet`](./resnet)__ is a setup for training ResNet18 on CIFAR-10 with simulated byzantine attackers
- __[`./albert`](./albert)__ runs distributed training of ALBERT-large with byzantine attacks using cloud instances

## ResNet18

This setup uses [torch.distributed](https://pytorch.org/docs/stable/distributed.html) for parallelism.

##### Requirements
- Python >= 3.7 (we recommend [Anaconda](https://www.anaconda.com/products/individual) python 3.8)
- Dependencies: `pip install jupyter torch>=1.6.0 torchvision>=0.7.0 tensorboard`
- A machine with at least 16GB RAM and either a GPU with >24GB memory or 3 GPUs with at least 10GB memory each.
- We tested the code on Ubuntu Server 18.04, it should work with all major linux distros. For Windows, we recommend using [Docker](https://www.docker.com/) (e.g. via [Kitematic](https://kitematic.com/)).

__Running experiments:__ please open __[`./resnet/RunExperiments.ipynb`](./resnet/RunExperiments.ipynb)__ and follow the instructions in that notebook.
The learning curves will be available in Tensorboard logs: `tensorboard --logdir btard/resnet`.

## ALBERT

This setup spawns distributed nodes that collectively train ALBERT-large on wikitext103. It uses a version of the [hivemind](https://github.com/learning-at-home/hivemind) library modified so that some peers may be programmed to become Byzantine and perform various types of attacks on the training process.

##### Requirements

- The experiments are optimized for 16 instances each with a single T4 GPU.
  - For your convenience, we provide a cost-optimized AWS starter notebook that can run experiments (see below)
  - While it can be simulated with a single node, doing so will require additional tuning depending on the number and type of GPUs available.

- If running manually, please install the core library on each machine:
  - The code requires python >= 3.7 (we recommend [Anaconda](https://www.anaconda.com/products/individual) python 3.8)
  - Install the library: __`cd ./albert/hivemind/ && pip install -e .`__
  - If successful, it should become available as `import hivemind`

__Running experiments:__ For your convenience, we provide a unified script that runs a distributed ALBERT experiment in the AWS cloud __[`./albert/experiments/RunExperiments.ipynb`](./albert/experiments/RunExperiments.ipynb)__ using preemptible T4 instances.
The learning curves will be posted to the Wandb project specified during the notebook setup.

__Expected cloud costs:__ a training experiment with 16 hosts takes up approximately $60 per day for g4dn.xlarge and $90 per day for g4dn.2xlarge instances. One can expect a full training experiment to converge in ≈3 days. Once the model is trained, one can restart training from intermediate checkpoints and simulate attacks. One attack episode takes up 4-5 hours depending on cloud availability.
