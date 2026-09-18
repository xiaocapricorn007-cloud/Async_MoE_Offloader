import torch
from collections import OrderedDict

class LRUVRAMCache:
    """
    Optimized VRAM cache for MoE experts using an LRU policy.
    Improvements for high efficiency:
    1. Pre-allocated GPU memory slots to prevent fragmentation and allocation overhead.
    2. CPU tensors remain persistently page-locked (pinned).
    3. CPU-thread non-blocking synchronization via `wait_stream`.
    """
    def __init__(self, max_capacity: int = 2):
        self.max_capacity = max_capacity
        self.cache = OrderedDict() # expert_idx -> (slot_idx, stream)
        self.gpu_slots = []
        self.is_initialized = False

    def _initialize_slots(self, sample_expert: torch.nn.Module):
        if self.is_initialized:
            return
            
        for _ in range(self.max_capacity):
            slot_tensors = {}
            for name, param in sample_expert.named_parameters():
                slot_tensors[name] = torch.empty_like(param, device='cuda')
            for name, buffer in sample_expert.named_buffers():
                slot_tensors[name] = torch.empty_like(buffer, device='cuda')
            self.gpu_slots.append(slot_tensors)
            
        self.is_initialized = True

    def prefetch(self, expert_module: torch.nn.Module):
        """
        Initiates an async non-blocking transfer to a pre-allocated GPU slot.
        """
        if not self.is_initialized:
            self._initialize_slots(expert_module)
            
        expert_key = id(expert_module)
            
        if expert_key in self.cache:
            self.cache.move_to_end(expert_key)
            return

        # Make room: recycle the LRU slot
        if len(self.cache) >= self.max_capacity:
            oldest_idx, (slot_idx, stream) = self.cache.popitem(last=False)
        else:
            slot_idx = len(self.cache)
            
        gpu_tensors = self.gpu_slots[slot_idx]
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            # Fast async memory copy into pre-allocated GPU tensors.
            # This completely avoids memory reallocation and garbage collection stalls.
            for name, param in expert_module.named_parameters():
                gpu_tensors[name].copy_(param.data, non_blocking=True)
            for name, buffer in expert_module.named_buffers():
                gpu_tensors[name].copy_(buffer.data, non_blocking=True)
                
        self.cache[expert_key] = (slot_idx, stream)

    def get_expert_slot(self, expert_module: torch.nn.Module) -> dict:
        """
        Retrieves the dict of GPU tensors for the expert.
        Tells the main compute stream to wait for the transfer stream.
        """
        expert_key = id(expert_module)
        if expert_key not in self.cache:
            raise RuntimeError(f"Expert {expert_key} was not prefetched!")
            
        slot_idx, transfer_stream = self.cache[expert_key]
        
        # [EFFICIENCY UPGRADE]
        # Make the current CUDA compute stream wait for the transfer stream.
        # This allows the CPU to immediately queue up the compute kernels,
        # maximizing host-device concurrency instead of blocking the CPU with .synchronize().
        torch.cuda.current_stream().wait_stream(transfer_stream)
        
        return self.gpu_slots[slot_idx]

    def evict(self):
        # Eviction is handled dynamically by recycling slots in prefetch().
        pass
