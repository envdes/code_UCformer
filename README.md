The repository is hosted anonymously on GitHub and will be made public upon paper acceptance.

### Description

- src: contains code of models used in this work
- config: describe the model configuration
- checkpoint: the checkpoints of `ucformer`,`mlp-csb`,`transformer`, and `automl` are hosted on hugging face, and the link goes [here](https://huggingface.co/XiGuaaa/ucformer)

- datasets: the datasets leveraged for model development are hosted on hugging face, and the link goes [here](https://huggingface.co/datasets/XiGuaaa/ucformer_dataset).

    The data catalog structure is as follows:

    - nc: this folder contains the `.nc` file for simulations, atmospheric forcings, and urban surface features.
    - Processed: this folder contains the `.parquet` file sourced from the corresponding `.nc` files., which are used for model development and evaluation.

