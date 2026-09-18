import torch.nn as nn
from .cache_manager import LRUVRAMCache
from .offloaded_block import OffloadedMoeBlock

def attach_lru_offloader(model: nn.Module, max_experts_in_vram: int = 64, _global_cache: LRUVRAMCache = None) -> nn.Module:
    """
    Recursively loop through model.named_modules().
    When a Qwen2MoeSparseMoeBlock is found, wrap the block in OffloadedMoeBlock,
    and use setattr to replace it. A single global cache is shared across all layers.
    """
    if _global_cache is None:
        _global_cache = LRUVRAMCache(max_capacity=max_experts_in_vram)
        
    for name, module in model.named_children():
        if module.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
            # Wrap the block with our custom offloading logic sharing the global cache
            offloaded_block = OffloadedMoeBlock(original_block=module, cache_manager=_global_cache)
            setattr(model, name, offloaded_block)
        else:
            attach_lru_offloader(module, max_experts_in_vram, _global_cache)
            
    return model
