import time
import numpy as np
from typing import Dict, Any, List
from vllm import LLM, SamplingParams

class VLLMProfiler:
    """Isolated profiler for real FP16 vs FP8 vLLM latency & throughput."""
    
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.85):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization

    def profile_format(
        self, 
        quant_format: str, 
        num_prompts: int = 100, 
        input_len: int = 512, 
        output_len: int = 128
    ) -> Dict[str, Any]:
        """Profiles throughput and latency for supported vLLM formats (fp16/bf16, fp8)."""
        print(f"\n[vLLM Track] Profiling Format: {quant_format.upper()}...")
        
        # Map benchmark config format to vLLM quantization string
        vllm_quant = "fp8" if quant_format.lower() == "fp8" else None
        
        # Generate dummy input tokens to eliminate network / dataset loader noise
        dummy_prompt_token_ids = [
            np.random.randint(100, 10000, size=input_len).tolist() 
            for _ in range(num_prompts)
        ]
        
        # Initialize vLLM Engine
        llm = LLM(
            model=self.model_name,
            quantization=vllm_quant,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False # Uses CUDA Graphs for optimal speed
        )
        
        sampling_params = SamplingParams(
            min_tokens=output_len,
            max_tokens=output_len,
            temperature=0.0
        )
        
        # Warmup pass
        _ = llm.generate(prompt_token_ids=dummy_prompt_token_ids[:2], sampling_params=sampling_params)
        
        # Timed Execution Pass
        start_time = time.perf_counter()
        outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids, sampling_params=sampling_params)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput_tps = total_generated_tokens / total_time
        
        print(f"  -> Processed {total_generated_tokens} tokens in {total_time:.2f}s ({throughput_tps:.2f} tokens/sec)")
        
        # Free GPU VRAM before next run
        del llm
        import torch
        torch.cuda.empty_cache()
        
        return {
            "format": quant_format,
            "tokens_per_second": round(throughput_tps, 2),
            "total_latency_sec": round(total_time, 2),
            "num_prompts": num_prompts,
            "generated_tokens": total_generated_tokens
        }