# Async MoE Offloader 🚀

Welcome to **Async MoE Offloader**, a highly optimized PyTorch interception layer designed to run massive Mixture of Experts (MoE) models (like `Qwen1.5-MoE-A2.7B`) on memory-constrained GPUs (e.g., 4GB VRAM).

## The Problem
Standard MoE models load all experts into the GPU, demanding massive amounts of VRAM. A typical 2.7B parameter MoE model requires at least 6-8GB of VRAM and crashes smaller GPUs with Out-Of-Memory (OOM) errors.

## The Solution
This package implements **Dynamic Asynchronous Expert Offloading**. It intercepts the MoE architecture to:
1. Keep the active router and core layers on the GPU.
2. Park the bulky MoE Experts in host CPU memory (RAM).
3. Dynamically swap only the needed experts into a constrained VRAM LRU (Least Recently Used) cache across the PCIe bus *only when routed to*.

### High-Performance Optimizations (Megatron-Style)
*   **Persistent Page-Locked (Pinned) Memory**: CPU-bound expert tensors are pinned for blazing-fast PCIe transfers.
*   **Pre-allocated Static VRAM Slots**: Eliminates Python-side `nn.Module` reallocation and completely prevents PyTorch caching allocator fragmentation. No `empty_cache()` stalls!
*   **Event-based Non-Blocking Synchronization**: Uses PyTorch's `wait_stream` on the GPU side, allowing the CPU host thread to dispatch compute kernels without stalling.
*   **Zero-Copy Pointer Swapping**: The forward pass dynamically re-routes underlying C++ tensor pointers instead of modifying `state_dict` dicts, saving critical micro-seconds.

## Directory Structure
- `moe_offloader/`: The core package containing the cache manager, patcher, and custom MoE blocks.
- `main.py`: The entrypoint for benchmarking and running generations.
- `PLAN.md`: Deep-dive into the architectural mechanics and memory management.
- `RUNNER.md`: Execution guide, system requirements, and telemetry explanation.

---
*Built for production-grade, asynchronous MoE inference on edge devices.*
