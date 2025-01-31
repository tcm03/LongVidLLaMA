from safetensors.torch import load_file

# Load the safetensors file
safetensors_path = "./checkpoints/longvu_qwen2/model.safetensors"
model = load_file(safetensors_path)

print(model['lm_head.weight'].shape)
print(model['model.embed_tokens.weight'].shape)


