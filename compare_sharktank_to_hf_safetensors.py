import safetensors.torch
from sharktank.utils import cli
parser = cli.create_parser()
parser.add_argument(
    "--sharktank_safetensors_file",
    help="sharktank safetensors file",
)
parser.add_argument(
    "--hf_safetensors_file",
    help="hf safetensors file",
)
args = cli.parse(parser)

sharktank_to_hf_layers = {
    "attn_norm" : "input_layernorm",
    "attn_q" : "self_attn.q_proj",
    "attn_k" : "self_attn.k_proj",
    "attn_v" : "self_attn.v_proj",
    "attn_output" : "self_attn.o_proj",
    "ffn_norm" : "post_attention_layernorm",
    "ffn_gate" : "mlp.gate_proj",
    "ffn_up" : "mlp.up_proj",
    "ffn_down" : "mlp.down_proj",
}


def find_matching_key(substring_key, keys):
    for key in keys:
        if substring_key in key:
            return key
    return None


def print_all_tensor_shapes(sharktank_file_path, hf_file_path):
    # Load the tensors from the safetensors file
    sharktank_tensors = safetensors.torch.load_file(sharktank_file_path)
    hf_tensors = safetensors.torch.load_file(hf_file_path)

    for sharktank_key, hf_key in sharktank_to_hf_layers.items():
        sharktank_key = find_matching_key(sharktank_key, sharktank_tensors.keys())
        hf_key = find_matching_key(hf_key, hf_tensors.keys())

        if sharktank_key and hf_key:
            sharktank_tensor_shape = sharktank_tensors[sharktank_key].shape
            hf_tensor_shape = hf_tensors[hf_key].shape
            print(f"'{sharktank_key}': {sharktank_tensor_shape}, '{hf_key}': {hf_tensor_shape}")
        else:
            if not sharktank_key:
                print(f"'{sharktank_key}' not found in sharktank safetensors file.")
            if not hf_key:
                print(f"'{hf_key}' not found in huggingface safetensors file.")


sharktank_file_path = args.sharktank_safetensors_file
hf_file_path = args.hf_safetensors_file

print_all_tensor_shapes(sharktank_file_path, hf_file_path)