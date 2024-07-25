from transformers import AutoTokenizer, LlamaForCausalLM
import transformers

model = LlamaForCausalLM.from_pretrained(
    "/srv/shark/Llama-2-70b-chat-hf"
)
tokenizer = AutoTokenizer.from_pretrained("/srv/shark/Llama-2-70b-chat-hf")

prompt = "What is MLIR?"
inputs = tokenizer(prompt, return_tensors="pt")
print('INPUTS:', inputs.input_ids)

generate_ids = model.generate(inputs.input_ids, max_length=30)
print('GENERATE IDS:', generate_ids, generate_ids.shape)

results = model.forward(inputs.input_ids)
print(results.logits)
print(results.logits.shape)

# print('DECODE:', tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])