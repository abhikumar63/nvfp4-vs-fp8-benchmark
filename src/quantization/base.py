from abc import ABC, abstractmethod
from typing import Tuple
import torch
import torch.nn as nn


class QuantizerEngine(ABC):
    """Abstract Base Class for model quantization backends."""

    def __init__(self, model_name_or_path: str, device: str = "cpu", torch_dtype: str = "bfloat16"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = self._parse_dtype(torch_dtype)

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(dtype_str, torch.float32 if self.device == "cpu" else torch.bfloat16)

    @abstractmethod
    def load_and_quantize(self) -> Tuple[nn.Module, object]:
        """Loads model and tokenizer, applying the backend's quantization scheme.

        Returns:
            Tuple[nn.Module, PreTrainedTokenizer]: Quantized model and tokenizer.
        """
        pass