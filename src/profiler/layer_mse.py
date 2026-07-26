from typing import Dict, List
import torch
import torch.nn as nn


class LayerWiseProfiler:
    """Attaches forward hooks to Transformer layers to calculate activation MSE vs. FP16 baseline."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def register_hooks(self):
        """Attaches hooks to all Transformer Block modules."""
        for name, module in self.model.named_modules():
            # Matches standard Transformer block naming conventions (Llama, Qwen, OPT)
            if any(key in name for key in ["layers.", "blocks.", "h."]) and len(name.split(".")) <= 3:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name: str):
        def hook(module, input, output):
            # Output can be a tuple (hidden_states, att_weights)
            tensor_output = output[0] if isinstance(output, tuple) else output
            self.activations[name] = tensor_output.detach().cpu().float()
        return hook

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @staticmethod
    def compare_activations(base_act: Dict[str, torch.Tensor], quant_act: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Computes Mean Squared Error (MSE) per layer."""
        layer_mse = {}
        for name in base_act:
            if name in quant_act:
                mse = torch.mean((base_act[name] - quant_act[name]) ** 2).item()
                layer_mse[name] = float(mse)
        return layer_mse