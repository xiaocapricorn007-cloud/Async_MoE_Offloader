# Runner Guide 🏎️

This guide explains how to execute the benchmarking script (`main.py`) to verify the VRAM constraints and test the text generation throughput.

## System Requirements
- OS: Linux / Windows
- Python: 3.9+
- Packages:
  ```bash
  pip install torch transformers accelerate bitsandbytes auto-gptq
  ```
- Hardware: At least 16GB CPU RAM and a GPU with at least 4GB VRAM.

## Execution
Run the benchmark script from the root of the repository:
```bash
python main.py
```

## What Happens Under The Hood?
1. **Model Loading:** The `Qwen1.5-MoE` GPTQ model is loaded entirely into CPU RAM (the parking lot).
2. **Patching:** The `attach_lru_offloader` monkey-patches the huggingface architecture. It scans the model tree and wraps standard MoE layers inside our custom `OffloadedMoeBlock`.
3. **Core Transfer:** The non-expert layers (Embeddings, Attention, LM Head) are moved to the GPU since they must be active for every token.
4. **Generation:** A single generation pass is triggered. The terminal will stream the dynamic VRAM usage.

## Interpreting the Telemetry Output
At the end of execution, you will see telemetry similar to this:
```text
--- Telemetry ---
Tokens generated: 50
Generation time: X.XX seconds
Throughput: Y.YY tokens/sec
[End of Generation] VRAM Allocated: 3.21 GB | Reserved: 3.40 GB
```
*   **VRAM Allocated:** The actual memory holding tensors. This MUST stay under the 3.5 GB constraint.
*   **Throughput:** The speed of generation. Due to the PCIe bandwidth bottleneck, this will be slower than native GPU execution, but it allows the model to run without OOM crashing.
