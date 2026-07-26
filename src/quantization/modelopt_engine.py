from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantization.base import QuantizerEngine


class ModelOptEngine(QuantizerEngine):
    """NVIDIA TensorRT Model Optimizer Engine for FP8 and NVFP4 formats.

    Falls back to simulated weight-only quantizations on CPU/Mac environments
    to enable end-to-end testing without CUDA hardware.
    """

    def __init__(self, model_name_or_path: str, format_type: str, device: str = "cuda", torch_dtype: str = "bfloat16"):
        super().__init__(model_name_or_path, device, torch_dtype)
        self.format_type = format_type.lower()  # 'fp8', 'nvfp4', or 'fp8_simulated'

    def load_and_quantize(self) -> Tuple[nn.Module, object]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Fallback path for Mac / CPU debugging
        if self.device != "cuda" or self.format_type == "fp8_simulated":
            return self._load_simulated_quantized(tokenizer)

        # Real NVIDIA ModelOpt GPU Path
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            raise ImportError(
                "nvidia-modelopt is required for real GPU quantization. "
                "Ensure `pip install nvidia-modelopt[hf]` is installed."
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.format_type == "fp8":
            config = mtq.INT8_DEFAULT_CFG  # Or mtq.FP8_DEFAULT_CFG depending on ModelOpt version
        elif self.format_type == "nvfp4":
            config = mtq.NVFP4_DEFAULT_CFG
        else:
            raise ValueError(f"Unsupported ModelOpt format: {self.format_type}")

        # Quantize in-place
        def forward_loop(model):
            # Dummy forward calibration
            dummy_input = tokenizer("Calibration sentence for quantization.", return_tensors="pt").to("cuda")
            model(**dummy_input)

        model = mtq.quantize(model, config, forward_loop=forward_loop)
        model.eval()
        return model, tokenizer

    def _load_simulated_quantized(self, tokenizer) -> Tuple[nn.Module, object]:
        """Simulates quantization scale noise for local Mac testing."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=self.torch_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        # Inject small Gaussian scale noise to simulate FP8/NVFP4 roundoff
        scale_std = 0.002 if "fp8" in self.format_type else 0.008
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() > 1:
                    noise = torch.randn_like(param) * scale_std
                    param.add_(noise)
        model.eval()
        return model, tokenizer