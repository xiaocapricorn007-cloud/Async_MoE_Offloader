import torch
import torch.nn as nn
from .cache_manager import LRUVRAMCache

class OffloadedMoeBlock(nn.Module):
    """
    A custom MoE Block that replaces the standard Qwen2MoeSparseMoeBlock.
    It keeps the router on the GPU, but dynamically fetches experts from CPU to GPU.
    """
    def __init__(self, original_block: nn.Module, cache_manager: LRUVRAMCache):
        super().__init__()
        # Preserve original components
        self.gate = original_block.gate
        self.experts = original_block.experts
        self.cache_manager = cache_manager
        
        # Preserve relevant attributes
        self.hidden_dim = getattr(original_block, 'hidden_dim', None)
        self.top_k = getattr(original_block, 'top_k', 2)
        
        # 1. Page-Locked Memory (Pinned Memory)
        # Pin the CPU-bound expert tensors upon initialization so that 
        # non-blocking PCIe transfers work efficiently during the forward pass.
        for expert in self.experts:
            for param in expert.parameters():
                if param.device.type == 'cpu':
                    param.data = param.data.pin_memory()
            for buffer in expert.buffers():
                if buffer.device.type == 'cpu':
                    buffer.data = buffer.data.pin_memory()

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 1. Pass hidden_states through the gate (Router) to get routing_weights and selected_experts
        router_logits = self.gate(hidden_states_flat)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize weights
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Get unique experts required for this batch
        unique_experts = selected_experts.unique().tolist()
        
        # 2. Immediately loop through selected_experts and call cache_manager.prefetch()
        # This starts all PCIe bus transfers concurrently on separate streams.
        for expert_idx in unique_experts:
            self.cache_manager.prefetch(expert_idx, self.experts[expert_idx])
            
        # 3. Loop through them again, pointer-swap, compute, and accumulate
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        
        for expert_idx in unique_experts:
            # Retrieves pre-allocated GPU tensors and queues a wait_stream on the GPU
            gpu_tensors = self.cache_manager.get_expert_slot(expert_idx)
            expert = self.experts[expert_idx]
            
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)
            token_idx, top_k_idx = torch.where(expert_mask)
            
            if token_idx.shape[0] == 0:
                continue
                
            # Extract the specific tokens and their routing weights
            tokens_for_expert = hidden_states_flat[token_idx]
            weights_for_expert = routing_weights[token_idx, top_k_idx].unsqueeze(-1)
            
            # [EFFICIENCY UPGRADE]: Pointer Swapping
            # Temporarily point the expert's parameters to the pre-allocated GPU memory.
            # This completely avoids instantiating copies of nn.Module or modifying the 
            # state_dict repeatedly, saving valuable micro-seconds.
            cpu_tensors = {}
            for name, param in expert.named_parameters():
                cpu_tensors[name] = param.data
                param.data = gpu_tensors[name]
            for name, buffer in expert.named_buffers():
                cpu_tensors[name] = buffer.data
                buffer.data = gpu_tensors[name]
                
            # Compute expert output (kernels are queued to the main stream asynchronously)
            expert_out = expert(tokens_for_expert)
            
            # Restore CPU pointers to keep the pinned "parking lot" intact
            for name, param in expert.named_parameters():
                param.data = cpu_tensors[name]
            for name, buffer in expert.named_buffers():
                buffer.data = cpu_tensors[name]
            
            # Accumulate the scaled results
            final_hidden_states[token_idx] += expert_out * weights_for_expert
            
        # Reshape back to original dimensions
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        
        # Standard huggingface MoE returns (hidden_states, router_logits)
        return final_hidden_states, router_logits
