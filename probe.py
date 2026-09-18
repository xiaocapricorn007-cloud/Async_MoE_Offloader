import torch
from transformers import AutoModelForCausalLM, GPTQConfig

def probe_model():
    gptq_config = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        device_map="cpu",
        quantization_config=gptq_config,
        trust_remote_code=True
    )
    
    print("\n--- MOE BLOCK STRUCTURE ---")
    print(model.model.layers[0].mlp)
    
if __name__ == "__main__":
    probe_model()
