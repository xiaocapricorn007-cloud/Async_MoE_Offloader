import torch.nn as nn
from .cache_manager import LRUVRAMCache
from .offloaded_block import OffloadedMoeBlock

def attach_lru_offloader(model: nn.Module, max_experts_in_vram: int = 2) -> nn.Module:
    """
    Recursively loop through model.named_modules().
    When a Qwen2MoeSparseMoeBlock is found, instantiate an LRUVRAMCache for that layer,
    wrap the block in OffloadedMoeBlock, and use setattr to replace it.
    """
    for name, module in model.named_children():
        # Check by class name to avoid direct import dependencies 
        # which might break if the transformers library version changes.
        if module.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
            # Create the VRAM LRU Cache for this specific MoE layer
            cache_manager = LRUVRAMCache(max_capacity=max_experts_in_vram)
            
            # Wrap the block with our custom offloading logic
            offloaded_block = OffloadedMoeBlock(original_block=module, cache_manager=cache_manager)
            
            # Replace the original block
            setattr(model, name, offloaded_block)
        else:
            # Recursively search in child modules
            attach_lru_offloader(module, max_experts_in_vram)
            
    return model
