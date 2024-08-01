from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
from ..utils.tokenizer import InferenceTokenizer
import torch


from ..utils import cli

parser = cli.create_parser()
parser.add_argument("prompt", nargs="+", help="Prompt strings")
parser.add_argument(
    "--save_intermediates_path",
    help="save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors",
)
cli.add_tokenizer_options(parser)
args = cli.parse(parser)
tokenizer = cli.get_tokenizer(args)

model = LlamaForCausalLM.from_pretrained("/home/avsharma/Llama-2-70b-chat-hf")
if args.save_intermediates_path:
    from ..utils.patching import SaveModuleResultTensorsPatch

    intermediates_saver = SaveModuleResultTensorsPatch()
    intermediates_saver.patch_child_modules(model)

prompts = args.prompt
# tokenizer = AutoTokenizer.from_pretrained("/srv/shark/Llama-2-70b-chat-hf")
token_ids, seq_lens = tokenizer.encode(prompts, pad_to_multiple_of=16)
token_ids = torch.tensor(token_ids)

# inputs = tokenizer(prompt, return_tensors="pt")
# print('INPUTS:', inputs.input_ids)

# results = model.forward(inputs.input_ids)
results = model.forward(token_ids)
if args.save_intermediates_path:
    intermediates_saver.save_file(args.save_intermediates_path + "_prefill.safetensors")
import numpy as np

np.save("test_llama_results.npy", results.logits.detach().numpy())
# print(results.logits)
# print(results.logits.shape)

# print('DECODE:', tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
