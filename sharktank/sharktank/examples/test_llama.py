from transformers import AutoTokenizer, LlamaForCausalLM
import transformers


from ..utils import cli
parser = cli.create_parser()
parser.add_argument(
    "--save_intermediates_path",
    help="save module forward outputs to safetensors, ex: run_0 will save to run_0_prefill.savetensors",
)
args = cli.parse(parser)

model = LlamaForCausalLM.from_pretrained(
    "/srv/shark/Llama-2-70b-chat-hf"
)
if args.save_intermediates_path:
    from ..utils.patching import SaveModuleResultTensorsPatch

    intermediates_saver = SaveModuleResultTensorsPatch()
    intermediates_saver.patch_child_modules(model)
tokenizer = AutoTokenizer.from_pretrained("/srv/shark/Llama-2-70b-chat-hf")

prompt = "What is MLIR?"
inputs = tokenizer(prompt, return_tensors="pt")
print('INPUTS:', inputs.input_ids)

# generate_ids = model.generate(inputs.input_ids, max_length=30)
# print('GENERATE IDS:', generate_ids, generate_ids.shape)

results = model.forward(inputs.input_ids)
if args.save_intermediates_path:
    intermediates_saver.save_file(
        args.save_intermediates_path + "_prefill.safetensors"
    )
# print(results.logits)
# print(results.logits.shape)

# print('DECODE:', tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])