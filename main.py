import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from moe_offloader.patcher import attach_lru_offloader
from safety_monitor import SafetyWatchdog

def print_memory_profile(tag: str):
    """Utility to print current CUDA VRAM usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{tag}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def main():
    # Safety Watchdog set to exactly 99.5% System RAM and 3.9GB VRAM
    watchdog = SafetyWatchdog(vram_limit_gb=3.9, ram_percent_limit=99.5, check_interval=0.1)
    
    model_id = "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4"
    print(f"Loading {model_id} into CPU RAM...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    # GPTQ models by default use the Exllama backend which strictly requires all weights
    # to be loaded directly to the GPU. Since our architecture relies on loading to the CPU
    # parking lot first, we must disable Exllama during initialization.
    gptq_config = GPTQConfig(bits=4, disable_exllama=True)
        
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            quantization_config=gptq_config,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Loading failed: {e}")
        return

    print_memory_profile("Post-Load CPU")

    print("\nAttaching Dynamic Asynchronous Expert Offloader...")
    # Monkey-Patch the model
    # Adjusted max_experts_in_vram to 330
    model = attach_lru_offloader(model, max_experts_in_vram=330)

    print("\nMoving core components (Embeddings, LM Head, Routers) to GPU...")
    # We must move the non-expert parts to the GPU for execution.
    # The OffloadedMoeBlock will handle keeping experts on CPU and fetching them dynamically.
    for name, param in model.named_parameters():
        if "mlp.experts" not in name:
            param.data = param.data.to('cuda')
    for name, buffer in model.named_buffers():
        if "mlp.experts" not in name:
            buffer.data = buffer.data.to('cuda')
            
    # Start the watchdog ONLY after the model is safely loaded into memory
    watchdog.start()

    print_memory_profile("Post-Core-Transfer")

    # Benchmarking / Testing task
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always reply in English."},
        {"role": "user", "content": "Explain the theory of relativity in simple terms."}
    ]
    
    # Apply the proper chat formatting so the model knows it's a conversation
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to('cuda')

    print(f"\nGenerating response for prompt...")
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()
    generation_time = end_time - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = num_tokens / generation_time

    print("\n--- Output ---")
    # Safely print on Windows terminals (cp1252) to avoid UnicodeEncodeError crashes
    print(generated_text.encode('ascii', 'replace').decode('ascii'))
    print("--------------\n")

    print("--- Telemetry ---")
    print(f"Tokens generated: {num_tokens}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    print_memory_profile("End of Generation")
    
    # Assert VRAM stays under 3.5 GB constraint
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    if allocated_gb < 3.5:
        print("\nSUCCESS: VRAM footprint successfully constrained below 3.5 GB limit.")
    else:
        print("\nWARNING: VRAM footprint exceeded the 3.5 GB limit.")

    watchdog.stop()

if __name__ == "__main__":
    main()
