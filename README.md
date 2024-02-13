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
mlx-llm-server --model <path-to-your-model>
```
### Arguments
- `--model`: The path to the mlx model weights, tokenizer, and config. This argument is required.
- `--adapter-file`: (Optional) The path for the trained adapter weights.

### Host and Port Configuration
The server will start on the host and port specified by the environment variables `HOST` and `PORT`. If these are not set, it defaults to `127.0.0.1:8080`.

To start the server on a different host or port, set the `HOST` and `PORT` environment variables before starting the server. For example:

```bash
export HOST=0.0.0.0
export PORT=5000
mlx-llm-server --model <path-to-your-model>
```

The MLX-LLM server can serve both Hugging Face format models and quantized MLX models. You can find these models at the [MLX Community on Hugging Face](https://huggingface.co/mlx-community).

## API Spec
## API Endpoint: `/v1/chat/completions`
### Method: `POST`
### Request Headers
- `Content-Type`: Must be `application/json`.

### Request Body (JSON)
- `messages`: An array of message objects representing the conversation history. Each message object should have a `role` (e.g., `user`, `assistant`) and `content` (the message text).
- `role_mapping`: (Optional) A dictionary to customize the role prefixes in the generated prompt. If not provided, default mappings are used.
- `stop`: (Optional) An array of strings or a single string representing stopping conditions for the generation. These are sequences of tokens where the generation should stop.
- `max_tokens`: (Optional) An integer specifying the maximum number of tokens to generate. Defaults to 100.
- `stream`: (Optional) A boolean indicating if the response should be streamed. If `true`, responses are sent as they are generated. Defaults to `false`.
- `model`: (Optional) A string specifying the model to use for generation. This is not utilized in the provided code but could be used for selecting among multiple models.
- `temperature`: (Optional) A float specifying the sampling temperature. Defaults to 1.0.
- `top_p`: (Optional) A float specifying the nucleus sampling parameter. Defaults to 1.0.
- `repetition_penalty`: Optional. Applies a penalty to repeated tokens.
- `repetition_context_size`: Optional. The size of the context window for applying repetition penalty.

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
