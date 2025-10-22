#### To facilitate code usage and demo access, this repository is under continuous development.

## Introduction

This repository is a supplement to the paper **"Learning Urban Climate Dynamics via Physics-Guided Urban Surface–Atmosphere Interactions"** accepted to NeurIPS 2025 main conference.

The objectives of this project are:

- Leverage data-dirven methods  to represent the interaction between urban surface and the atmopheric forcing.
- Incorpate **physical and domian knowledge** to models for enchanced modelling.
- Investigate the **generalization** and **multi-task** capabilities of physics-guide models and their potential as urban climate **foundation models**.

### Description

- src: contains code of models used in this work
- config: describe the model configuration
- checkpoint: the checkpoints of `ucformer`,`mlp-csb`,`transformer`, and `automl` are hosted on hugging face, and the link goes [here](https://huggingface.co/XiGuaaa/ucformer)

- datasets: the datasets leveraged for model development are hosted on hugging face, and the link goes [here](https://huggingface.co/datasets/XiGuaaa/ucformer_dataset).

    The data catalog structure is as follows:

    - nc: this folder contains the `.nc` file for simulations, atmospheric forcings, and urban surface features.
    - Processed: this folder contains the `.parquet` file sourced from the corresponding `.nc` files., which are used for model development and evaluation.

