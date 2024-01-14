from mlx_lm import load, generate

model, tokenizer = load("mistralai/Mistral-7B-v0.1")

response = generate(model, tokenizer, prompt="hello", verbose=True)