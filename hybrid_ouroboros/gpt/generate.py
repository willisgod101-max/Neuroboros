import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0, top_k=None, device='cpu'):
    """
    Generate text from the model.
    Args:
        model: The GPT model.
        tokenizer: The tokenizer (with encode and decode methods).
        prompt: The initial text to condition on.
        max_new_tokens: The maximum number of tokens to generate.
        temperature: Temperature for sampling (default 1.0).
        top_k: If set, only sample from the top k tokens (default None).
        device: Device to run on (default 'cpu').
    Returns:
        The generated text as a string.
    """
    model.eval()
    model.to(device)
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    # Generate tokens
    for _ in range(max_new_tokens):
        # If the sequence context is growing too long we must crop it at block_size.
        # Get the positional embedding table (we only need up to the model's block size)
        block_size = model.block_size
        if input_ids.size(1) > block_size:
            input_ids = input_ids[:, -block_size:]
        # Forward the model to get the logits for the index in the sequence
        with torch.no_grad():
            logits, _ = model(input_ids)
        # Pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # Optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # Apply softmax to convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample from the distribution
        input_ids_next = torch.multinomial(probs, num_samples=1)
        # Append sampled index to the running sequence and continue
        input_ids = torch.cat((input_ids, input_ids_next), dim=1)
    # Decode the generated tokens
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

if __name__ == '__main__':
    # Placeholder for testing
    print("Generate module for GPT")