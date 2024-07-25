import safetensors.torch
from sharktank.utils import cli
parser = cli.create_parser()
parser.add_argument(
    "--file",
    help="save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors",
)
args = cli.parse(parser)

# Function to print all tensor names in a safetensors file
def print_all_tensor_names(file_path):
    # Load the tensors from the safetensors file
    loaded_tensors = safetensors.torch.load_file(file_path)
    
    # Print all tensor names
    print("Tensor names in the safetensors file:")
    for tensor_name in loaded_tensors.keys():
        print(tensor_name)

# Example usage
file_path = args.file

print_all_tensor_names(file_path)