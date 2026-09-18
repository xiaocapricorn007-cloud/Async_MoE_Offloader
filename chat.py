import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from moe_offloader.patcher import attach_lru_offloader
from safety_monitor import SafetyWatchdog

def print_memory_profile(context=""):
    print(f"[{context}] VRAM Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB | Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

def main():
    # Safety Watchdog set to exactly 99.5% System RAM and 3.9GB VRAM
    watchdog = SafetyWatchdog(vram_limit_gb=3.9, ram_percent_limit=99.5, check_interval=0.1)
    
    model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
    
    print(f"Loading {model_id} into CPU RAM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Intentionally mapping the model to 'cpu' forces the layers to load into RAM initially.
    # Exllama must be disabled because it strictly expects weights to be on the GPU.
    gptq_config = GPTQConfig(bits=4, disable_exllama=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        quantization_config=gptq_config,
        trust_remote_code=True
    )

    print("\nAttaching Dynamic Asynchronous Expert Offloader...")
    model = attach_lru_offloader(model, max_experts_in_vram=330)

    print("\nMoving core components (Embeddings, LM Head, Routers) to GPU...")
    for name, param in model.named_parameters():
        if "mlp.experts" not in name:
            param.data = param.data.to('cuda')
    for name, buffer in model.named_buffers():
        if "mlp.experts" not in name:
            buffer.data = buffer.data.to('cuda')
            
    # Start the watchdog ONLY after the model is safely loaded into memory
    watchdog.start()

    print_memory_profile("Model Ready")
    print("\n========================================================")
    print("🤖 Qwen Asynchronous MoE Chat Interface Online")
    print("Type 'exit' or 'quit' to shut down the model.")
    print("========================================================\n")

    # Conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always reply in English and keep your answers coherent and helpful."}
    ]

    try:
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            # Apply the proper chat formatting
            text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_prompt, return_tensors="pt").to('cuda')

            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    pad_token_id=tokenizer.eos_token_id
                )

            end_time = time.time()
            generation_time = end_time - start_time
            
            # The generation outputs the entire prompt + generated text. We slice out just the new tokens.
            num_input_tokens = inputs.input_ids.shape[1]
            generated_token_ids = outputs[0][num_input_tokens:]
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            
            num_tokens = len(generated_token_ids)
            tokens_per_sec = num_tokens / generation_time

            print("\nQwen: ", end="")
            # Safely print on Windows terminals (cp1252) to avoid UnicodeEncodeError crashes
            print(generated_text.encode('ascii', 'replace').decode('ascii'))
            
            # Append the model's response to the conversation history
            messages.append({"role": "assistant", "content": generated_text})

            print("\n" + "-"*50)
            print(f"⚡ Tokens generated: {num_tokens}")
            print(f"⏱️ Generation time: {generation_time:.2f} seconds")
            print(f"🚀 Throughput: {tokens_per_sec:.2f} tokens/sec")
            print_memory_profile("Current Footprint")
            print("-"  * 50 + "\n")

    except KeyboardInterrupt:
        print("\n[Shutting down model...]")
    finally:
        watchdog.stop()

if __name__ == "__main__":
    main()
