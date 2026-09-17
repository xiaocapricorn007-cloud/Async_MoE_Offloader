# Architectural Plan & Mechanics 🧠

This document outlines the engineering blueprint behind the Async MoE Offloader.

## Core Constraint
- **Host Memory (CPU RAM):** The "parking lot" containing all MoE Experts.
- **Device Memory (GPU VRAM):** The constrained execution cache (configurable limit, e.g., max 2 experts at a time).

## Mechanics

### 1. LRU VRAM Cache Manager (`cache_manager.py`)
Instead of dynamically creating and destroying Python `nn.Module` objects, we use a static memory pool:
- Upon initialization, the cache manager allocates `N` exact replicas of the expert architecture directly on the GPU (where `N` is `max_capacity`).
- These are just empty memory buffers (slots).

### 2. Pinned Memory `pin_memory()`
CPU memory is paged by the OS. The GPU cannot safely perform Direct Memory Access (DMA) on paged memory. By pinning the CPU tensors during the initialization of `OffloadedMoeBlock`, we lock them into physical RAM, enabling PyTorch's `non_blocking=True` to execute asynchronous DMA transfers over the PCIe bus.

### 3. Asynchronous Prefetching & `wait_stream`
When the router selects experts (e.g., Expert 3 and Expert 7):
1. **Prefetching:** A dedicated `torch.cuda.Stream()` instantly issues an asynchronous `H2D` copy from the pinned CPU memory into an available GPU buffer slot.
2. **Execution:** Instead of using `.synchronize()` which blocks the CPU, we use `torch.cuda.current_stream().wait_stream(transfer_stream)`. The CPU instantly queues the compute kernel on the main stream. The GPU's hardware scheduler guarantees the compute kernel will not execute until the `H2D` transfer stream event completes.

### 4. Zero-Overhead Pointer Swapping
Instead of modifying the `nn.Module` object (which allocates memory and drops references), the `OffloadedMoeBlock` hot-swaps the `.data` pointer of the CPU expert to point temporarily to the GPU buffer. The kernel executes using the GPU memory, and the pointer is immediately reverted.
