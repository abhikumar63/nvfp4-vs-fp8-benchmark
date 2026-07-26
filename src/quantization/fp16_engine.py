from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.quantization.base import QuantizerEngine


class FP16Engine(QuantizerEngine):
    """Baseline unquantized engine (FP16 / BF16)."""

    def load_and_quantize(self) -> Tuple[nn.Module, object]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            model = model.to("cpu")
            
        model.eval()
        return model, tokenizer