import os
import json
import argparse
from huggingface_hub import snapshot_download, HfApi

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Convert and upload a model to Hugging Face Hub.")
parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token")
parser.add_argument("--model_id", type=str, required=True, help="Model ID on Hugging Face Hub")
parser.add_argument("--outtype", type=str, required=True, help="Output type for the conversion (e.g., q8_0, f16, f32)")
args = parser.parse_args()

hf_token = args.hf_token
model_id = args.model_id
outtype = args.outtype

# Install necessary packages
os.system("pip install huggingface_hub")

# Define the local directory based on the model_id
local_dir = f"../models/{model_id}"

# Download the model snapshot
snapshot_download(repo_id=model_id, local_dir=local_dir, revision="main")

# Clone the llama.cpp repository and install its requirements
os.system("git clone https://github.com/ggerganov/llama.cpp.git")
os.system("pip install -r llama.cpp/requirements.txt")

# Modify the convert-hf-to-gguf.py script to handle T5 configuration
convert_script_path = "llama.cpp/convert-hf-to-gguf.py"
with open(convert_script_path, "r") as file:
    script_content = file.read()

# Add the modified loadHFTransformerJson function
modified_function = """
import json

class Params:
    def __init__(self, n_embd, n_layer, n_head, **kwargs):
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        # Add other necessary parameters

    @staticmethod
    def loadHFTransformerJson(model, hf_config_path):
        with open(hf_config_path, "r") as f:
            config = json.load(f)

        # Adjust the parameters based on the T5 configuration
        n_embd = config.get("hidden_size", config.get("d_model"))
        n_layer = config.get("num_hidden_layers", config.get("num_layers"))
        n_head = config.get("num_attention_heads", config.get("num_heads"))

        # Continue with the rest of the script
        params = Params(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            # Add other necessary parameters
        )

        return params

    @staticmethod
    def load(model_plus):
        hf_config_path = model_plus.model.config_path
        return Params.loadHFTransformerJson(model_plus.model, hf_config_path)

def main():
    # Your main function code here
    pass

if __name__ == "__main__":
    main()
"""

# Replace the original function in the script
script_content = script_content.replace(
    "def loadHFTransformerJson(model, hf_config_path):",
    modified_function
)

with open(convert_script_path, "w") as file:
    file.write(script_content)

# Convert the model using the specified outtype
conversion_command = f"python3 llama.cpp/convert-hf-to-gguf.py {local_dir} --outfile {local_dir}.gguf --outtype {outtype}"
conversion_result = os.system(conversion_command)

# Check if the conversion was successful
if conversion_result != 0:
    raise RuntimeError("Model conversion failed. Please check the convert-hf-to-gguf.py script for errors.")

# Check if the output file exists
if not os.path.isfile(f"{local_dir}.gguf"):
    raise FileNotFoundError(f"The converted model file '{local_dir}.gguf' was not found.")

# Upload the converted model to Hugging Face Hub
####

# api = HfApi(token=hf_token)
# api.create_repo(model_id, exist_ok=True, repo_type="model")
# api.upload_file(
#     path_or_fileobj=f"{local_dir}.gguf",
#     path_in_repo=f"{model_id}.gguf",
#     repo_id=model_id,
# )

# Review note
print(f"In this case we're also quantizing the model to 8 bit by setting --outtype {outtype}. "
      "Quantizing helps improve inference speed, but it can negatively impact quality. "
      "You can use --outtype f16 (16 bit) or --outtype f32 (32 bit) to preserve original quality.")
