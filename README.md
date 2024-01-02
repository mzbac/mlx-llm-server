# mlx-llm

## Overview

This repository provides examples for running LLM models in the HF (Hugging Face) format using the MLX framework. It expands upon the existing `mlx-examples` repository, which primarily focuses on models in the PyTorch format.

## Installation

### Setting Up Miniconda

For users with Apple Silicon hardware, it is recommended to install Miniconda natively. This can be done using the following commands:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
``````
### Creating a Conda Environment
After installing Miniconda, create a dedicated conda environment for mlx-llm:

```
conda create -n mlx-llm python=3.10
conda activate mlx-llm
```

### Installing Dependencies

To install the necessary dependencies, run the following command within the created conda environment:

```
pip install -r requirements.txt
```


## Usage

### Converting Hugging Face Models to MLX Format

To convert a model from the Hugging Face format to the MLX format, use:

```
python convert_llama_from_hf.py --hf-path <path_to_huggingface_model> --mlx-path <path_to_save_mlx_model>
```
For additional command-line options and details, refer to the help:

```
python convert_llama_from_hf.py --help
```

### Downloading Models from the Hugging Face Hub

To download models from the Hugging Face hub:

```
python download.py --model-name <model_name>
```
Note: the model will be downloaded to the `models` directory

### Running Inference on MLX Models

To execute inference using an MLX model, the following command can be used:
```
python inference.py --model-path <path_to_mlx_model> --prompt <prompt>
```

For more information on available options:

```
python inference.py --help
```

## Example: Using `mzbac/mlx-deepseek-coder-6.7b-instruct-4-bit`

This section provides a step-by-step guide on how to download and run the `mzbac/mlx-deepseek-coder-6.7b-instruct-4-bit` model.

1. **Download the Model**:

```
python download.py mzbac/mlx-deepseek-coder-6.7b-instruct-4-bit
```

2. **Run the Model**:

```
python inference.py --model-path models/mzbac/mlx-deepseek-coder-6.7b-instruct-4-bit --prompt "### Instruction: \nwrite a quick sort algorithm in python.\n### Response: \n"
```
** Note: ** More converted mlx models can be found [here](https://huggingface.co/mzbac).