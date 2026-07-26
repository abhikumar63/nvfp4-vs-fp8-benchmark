import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm


class PerplexityEvaluator:
    """Calculates Cross-Entropy Loss and Perplexity on WikiText-2 or C4."""

    def __init__(self, model: nn.Module, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataset_name: str = "Salesforce/wikitext", dataset_config: str = "wikitext-2-raw-v1", max_samples: int = 50, seq_len: int = 512) -> dict:
        dataset = load_dataset(dataset_name, dataset_config, split="test")
        text = "\n\n".join([s["text"] for s in dataset if len(s["text"].strip()) > 0])
        
        if max_samples:
            text = text[: max_samples * seq_len]

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids[0]
        
        nlls = []
        num_tokens = 0

        for i in tqdm(range(0, input_ids.size(0) - seq_len, seq_len), desc=f"Evaluating PPL ({dataset_name})"):
            batch_inputs = input_ids[i : i + seq_len].unsqueeze(0).to(self.device)
            target_ids = batch_inputs.clone()

            outputs = self.model(batch_inputs, labels=target_ids)
            neg_log_likelihood = outputs.loss * seq_len

            nlls.append(neg_log_likelihood)
            num_tokens += seq_len

        if not nlls:
            return {"loss": float("nan"), "perplexity": float("nan")}

        total_loss = torch.stack(nlls).sum() / num_tokens
        perplexity = torch.exp(total_loss).item()

        return {
            "loss": float(total_loss.item()),
            "perplexity": float(perplexity),
        }