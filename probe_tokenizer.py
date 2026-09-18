from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "who are you?"}
]

# Case 1: tokenize=False then tokenizer()
text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs_1 = tokenizer(text_prompt, return_tensors="pt")

# Case 2: tokenize=True
inputs_2 = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

print("Case 1 IDs:", inputs_1["input_ids"][0][:10])
print("Case 2 IDs:", inputs_2[0][:10])
