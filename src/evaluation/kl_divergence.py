import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitDriftEvaluator:
    """Measures KL-Divergence and Cosine Similarity of output logits relative to FP16 baseline."""

    @staticmethod
    @torch.no_grad()
    def compute_drift(base_logits: torch.Tensor, quant_logits: torch.Tensor) -> dict:
        """
        Args:
            base_logits: [batch_size, seq_len, vocab_size] from FP16 baseline
            quant_logits: [batch_size, seq_len, vocab_size] from Quantized model
        """
        # Temperature scaled log-probabilities
        p_base = F.log_softmax(base_logits, dim=-1)
        q_quant = F.log_softmax(quant_logits, dim=-1)

        # Forward KL-Divergence: KL(P_base || Q_quant)
        kl_div = F.kl_div(q_quant, p_base, log_target=True, reduction="batchmean").item()

        # Cosine Similarity across vocabulary dimension
        cos_sim = F.cosine_similarity(base_logits.view(-1, base_logits.size(-1)), 
                                     quant_logits.view(-1, quant_logits.size(-1)), 
                                     dim=-1).mean().item()

        return {
            "kl_divergence": float(kl_div),
            "cosine_similarity": float(cos_sim),
        }