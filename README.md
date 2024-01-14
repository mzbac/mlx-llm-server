# MLX-LLM

This guide will help you set up the MLX-LLM server to serve the model as an OpenAI compatible API.

## Quick Start

1. Start the server with the following command:

```bash
python -m server --model-path <path-to-your-model>
```

Replace <path-to-your-model> with the actual path to your MLX-LLM model. This will start the server and expose the MLX-LLM model as an API.

## Setup Guide
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
"messages": [
{
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about python exceptions"
}
]
}'
```

