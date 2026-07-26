import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def plot_summary_metrics(results: dict, output_dir: str):
    """Plots Perplexity (Lower is Better) and Throughput (Higher is Better)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Quantization Benchmark: {results.get('model', 'Unknown Model')}", fontsize=16, fontweight='bold')

    # --- Plot 1: Perplexity (Track A) ---
    acc_track = results.get("accuracy_track", {})
    if acc_track:
        formats_acc = list(acc_track.keys())
        ppl_values = [acc_track[f].get("perplexity", 0) for f in formats_acc]
        
        sns.barplot(x=formats_acc, y=ppl_values, ax=ax1, palette="Blues_d")
        ax1.set_title("Language Modeling Perplexity (Lower is Better)")
        ax1.set_ylabel("Perplexity on WikiText")
        
        # Annotate values
        for i, v in enumerate(ppl_values):
            ax1.text(i, v + 0.05, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

    # --- Plot 2: vLLM Throughput (Track B) ---
    vllm_track = results.get("vllm_track", {})
    if vllm_track:
        formats_vllm = list(vllm_track.keys())
        tps_values = [vllm_track[f].get("tokens_per_second", 0) for f in formats_vllm]
        
        sns.barplot(x=formats_vllm, y=tps_values, ax=ax2, palette="Greens_d")
        ax2.set_title("Inference Throughput (Higher is Better)")
        ax2.set_ylabel("Tokens per Second (TPS)")
        
        # Annotate values
        for i, v in enumerate(tps_values):
            ax2.text(i, v + 2, f"{v:.1f}", ha='center', va='bottom', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "vLLM Benchmark Disabled", ha='center', va='center', fontsize=12)
        ax2.set_title("Inference Throughput")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "benchmark_summary.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary chart to: {out_path}")
    plt.close()

def plot_layerwise_mse(results: dict, output_dir: str):
    """Plots the Mean Squared Error across Transformer layers."""
    acc_track = results.get("accuracy_track", {})
    
    # We only plot MSE for quantized formats compared to the FP16 baseline
    quant_formats = [f for f in acc_track.keys() if f != "fp16"]
    
    if not quant_formats or "mse_per_layer" not in acc_track[quant_formats[0]]:
        print("No layer-wise MSE data found. Skipping MSE plot.")
        return

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    for fmt in quant_formats:
        mse_data = acc_track[fmt].get("mse_per_layer", [])
        if mse_data:
            # Assuming mse_data is a list of floats corresponding to layer indices
            layers = list(range(len(mse_data)))
            plt.plot(layers, mse_data, marker='o', label=fmt.upper(), linewidth=2)

    plt.title("Layer-wise Activation MSE vs FP16 Baseline", fontsize=14, fontweight='bold')
    plt.xlabel("Transformer Layer Index")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.yscale("log") # Log scale helps visualize tiny errors effectively
    plt.legend()
    
    out_path = os.path.join(output_dir, "layer_mse_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved Layer MSE chart to: {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="outputs/llama_3.1_8b/benchmark_results.json")
    parser.add_argument("--outdir", type=str, default="outputs/llama_3.1_8b")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Could not find results file at {args.results}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    results = load_results(args.results)
    
    print("Generating visual reports...")
    plot_summary_metrics(results, args.outdir)
    plot_layerwise_mse(results, args.outdir)

if __name__ == "__main__":
    main()