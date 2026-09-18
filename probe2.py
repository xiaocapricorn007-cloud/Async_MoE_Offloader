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
    
    expert = model.model.layers[0].mlp.experts[0].down_proj
    
    print("\n--- ATTRIBUTES ---")
    for k, v in vars(expert).items():
        if isinstance(v, torch.Tensor):
            print(f"Tensor Attribute: {k} (shape: {v.shape})")
            
    print("\n--- PARAMETERS ---")
    for name, param in expert.named_parameters():
        print(f"Parameter: {name} (shape: {param.shape})")
        
    print("\n--- BUFFERS ---")
    for name, buffer in expert.named_buffers():
        print(f"Buffer: {name} (shape: {buffer.shape})")
        
if __name__ == "__main__":
    probe_model()
