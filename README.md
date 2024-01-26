# mlx-llm-server

This guide will help you set up the MLX-LLM server to serve the model as an OpenAI compatible API.

## Quick Start

### Installation

Before starting the MLX-LLM server, install the server package from PyPI:

```bash
pip install mlx-llm-server
```
### Start the Server

```bash
mlx-llm-server --model-path <path-to-your-model>
```
### Arguments
- `--model-path`: The path to the mlx model weights, tokenizer, and config. This argument is required.
- `--adapter-file`: (Optional) The path for the trained adapter weights.

### Host and Port Configuration
The server will start on the host and port specified by the environment variables `HOST` and `PORT`. If these are not set, it defaults to `127.0.0.1:8080`.

To start the server on a different host or port, set the `HOST` and `PORT` environment variables before starting the server. For example:

```bash
export HOST=0.0.0.0
export PORT=5000
mlx-llm-server --model-path <path-to-your-model>
```

The MLX-LLM server can serve both Hugging Face format models and quantized MLX models. You can find these models at the [MLX Community on Hugging Face](https://huggingface.co/mlx-community).

## Development Setup Guide
### Miniconda Installation
For Apple Silicon users, install Miniconda natively with these commands:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

### Conda Environment Setup
After Miniconda installation, create a dedicated conda environment for MLX-LLM:
```
conda create -n mlx-llm python=3.10
conda activate mlx-llm
```
### Installing Dependencies

With the `mlx-llm` environment activated, install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Testing the API with curl

You can test the API using the `curl` command. Here's an example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no-key" \
  -d '{
  "model": "gpt-3.5-turbo",
  "stop":["<|im_end|>"],
  "messages": [
    {
      "role": "user",
      "content": "Write a limerick about python exceptions"
    }
  ]
}'
```
