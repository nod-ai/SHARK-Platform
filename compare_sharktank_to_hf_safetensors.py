import torch
import re
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
    # "output_lm_head" : "lm_head",
    # "attn_norm" : "input_layernorm",
    # "attn_q" : "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    # "attn_v" : "self_attn.v_proj",
    # "attn_output" : "self_attn.o_proj",
    # "ffn_norm" : "post_attention_layernorm",
    # "ffn_gate" : "mlp.gate_proj",
    # "ffn_up" : "mlp.up_proj",
    # "ffn_down" : "mlp.down_proj",
    # "output_norm" : "norm",
    # "token_embedding" : "embed_tokens",
}


def find_matching_key(string_key, dictionary):
    for key in dictionary.keys():
        if string_key in key:
            return key
    return None


def compare_all_tensors(sharktank_file_path, hf_file_path):
    # Load the tensors from the safetensors file
    sharktank_tensors = safetensors.torch.load_file(sharktank_file_path)
    hf_tensors = safetensors.torch.load_file(hf_file_path)

    for sharktank_key in sharktank_tensors:
        sharktank_pattern = r"^[^.]*\.([^.]*)\.(.*)"
        sharktank_layer_key = re.search(sharktank_pattern, sharktank_key)
        if sharktank_layer_key:
            layer_num = sharktank_layer_key.group(1)
            sharktank_layer_substring = sharktank_layer_key.group(2)
            if sharktank_layer_substring in sharktank_to_hf_layers:
                hf_substring_key = sharktank_to_hf_layers[sharktank_layer_substring]
                hf_key = "model.layers." + layer_num + "." + hf_substring_key

                if sharktank_key and hf_key:
                    sharktank_tensor = sharktank_tensors[sharktank_key]
                    hf_tensor = hf_tensors[hf_key]
                    sharktank_tensor_shape = sharktank_tensor.shape
                    hf_tensor_shape = hf_tensor.shape
                    assert sharktank_tensor_shape == hf_tensor_shape
                    all_close_res = torch.allclose(sharktank_tensor, hf_tensor)
                    if not all_close_res:
                        print(f"LAYER '{sharktank_key}' NOT EQUALS '{hf_key}'")
                        import numpy as np

                        np.save(f"{sharktank_key}.npy", sharktank_tensor)
                        np.save(f"{hf_key}.npy", hf_tensor)
        else:
            continue


sharktank_file_path = args.sharktank_safetensors_file
hf_file_path = args.hf_safetensors_file

compare_all_tensors(sharktank_file_path, hf_file_path)
