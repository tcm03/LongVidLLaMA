from safetensors.torch import load_file

# Load the safetensors file
safetensors_path = "./checkpoints/longvu_qwen2/model.safetensors"
checkpoint = load_file(safetensors_path)

# Print all keys in the checkpoint
keys = checkpoint.keys()
for key in keys:
    if "lm_head" in key or "score" in key:
        print(key)

