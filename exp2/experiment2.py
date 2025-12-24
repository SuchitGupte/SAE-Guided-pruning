import torch
import gc
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ID = "gemma-2-2b"
# MODEL_ID = "gemma-2-9b"
OUT_TYPE = "res"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Initializing on {DEVICE}...")

# Utility: Clear VRAM
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def get_active_features(model, sae, text_list, top_k=100):
    """
    Returns indices of the top_k most active features for a dataset.
    Handles multi-GPU model distribution safely.
    """
    total_acts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    
    # Correct hook formatting
    hook_name = f"blocks.{sae.cfg.hook_layer}.hook_resid_post"

    def hook_fn(activations, hook):
        # activations: [batch, seq, d_model] on layer's device (e.g. cuda:3)
        # Move to SAE device (e.g. cuda:0) before encoding
        acts = activations.to(sae.device)
        
        feature_acts = sae.encode(acts)
        # Sum over batch and sequence length
        total_acts.add_(feature_acts.sum(dim=(0,1)))
    # Tokenize and run
    tokens = model.to_tokens(text_list, truncate=True)[:, :128]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    
    return torch.topk(total_acts, top_k).indices

def get_task_data(task_name, n_samples=100):
    """Returns a list of text strings for a specific task."""
    texts = []
    
    if task_name == 'code':
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
        for item in ds.take(n_samples):
            texts.append(item['content'])
            
    elif task_name == 'math':
        ds = load_dataset("gsm8k", "main", split="train", streaming=True)
        for item in ds.take(n_samples):
            texts.append(item['question'] + "\n" + item['answer'])

    elif task_name == 'english':
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train", streaming=True)
        for item in ds.take(n_samples):
            texts.append(item['text'])

    else:
        raise ValueError(f"Unknown task name: {task_name}")

    return texts


def get_feature_sets(model, sae, task_texts, baseline_texts, k=3000):
    """Returns sets of feature indices for Task, Baseline, and their differences."""
    task_indices = set(get_active_features(model, sae, task_texts, top_k=k).tolist())
    base_indices = set(get_active_features(model, sae, baseline_texts, top_k=k).tolist())
    
    unique_task = task_indices - base_indices     # KEEP these
    shared = task_indices.intersection(base_indices) # KEEP these (grammar/structure)
    english_only = base_indices - task_indices    # PRUNE these (distractions)
    
    return unique_task, shared, english_only

# --- Experiment: Plotting the "Prunable Capacity" ---
def scan_prunable_features(task="code"):
    print(f"Scanning Prunable Features for {task}...")
    baseline_texts = get_task_data("english", n_samples=100)
    task_texts = get_task_data(task, n_samples=100)
    
    prunable_counts = []
    # layers = range(0, 26) 
    layers = range(0, 42) 
    
    for layer in layers:
        try:
            release = f"gemma-scope-2b-pt-{OUT_TYPE}-canonical" 
            sae_id = f"layer_{layer}/width_16k/canonical"

            # release = f"gemma-scope-9b-pt-{OUT_TYPE}-canonical"
            # sae_id = f"layer_{layer}/width_131k/canonical" 
            
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            sae.to(DEVICE)

            _, _, english_only = get_feature_sets(model, sae, task_texts, baseline_texts)
            
            count = len(english_only)
            print(f"Layer {layer}: {count} features are English-Only (Prunable)")
            prunable_counts.append(count)
            del sae
        except:
            prunable_counts.append(0)
            
    plt.figure(figsize=(10,4))
    plt.plot(layers, prunable_counts, marker='o', color='red', linestyle='--')
    plt.title(f"Prunable Features ('English-Only') for Task: {task}")
    plt.xlabel("Layer")
    plt.ylabel("Count of Prunable Features")
    plt.grid(True)
    plt.savefig(f"prunable_scan_{task}.png")
    plt.show()


def scan_prunable_features_multik(task="code", k_values=[1000, 3000, 5000]):
    print(f"Scanning Prunable Features for {task} with k={k_values}...")
    
    # 1. Prepare Data
    baseline_texts = get_task_data("english", n_samples=100)
    task_texts = get_task_data(task, n_samples=100)
    
    # Dictionary to store results: {1000: [], 3000: [], 5000: []}
    results = {k: [] for k in k_values}
    layers = range(0, 42) 
    
    # 2. Run Sweep
    for layer in layers:
        print(f"Processing Layer {layer}...")
        try:
            # Load SAE once per layer
            release = f"gemma-scope-2b-pt-{OUT_TYPE}-canonical" 
            sae_id = f"layer_{layer}/width_16k/canonical"

            # release = f"gemma-scope-9b-pt-{OUT_TYPE}-canonical"
            # sae_id = f"layer_{layer}/width_131k/canonical" 
            
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            sae.to(DEVICE)

            # Compute stats for each K value using the same loaded SAE
            for k in k_values:
                _, _, english_only = get_feature_sets(model, sae, task_texts, baseline_texts, k=k)
                count = len(english_only)
                results[k].append(count)
                
            del sae
            cleanup() # Ensure VRAM is freed
            
        except Exception as e:
            print(f"  Error at Layer {layer}: {e}")
            for k in k_values:
                results[k].append(0)

    # 3. Plotting (1 Row, 3 Columns)
    fig, axes = plt.subplots(1, len(k_values), figsize=(20, 5), sharey=True)
    if len(k_values) == 1: axes = [axes] # Handle single K case

    for i, k in enumerate(k_values):
        ax = axes[i]
        counts = results[k]
        
        # Plot styling
        ax.plot(layers, counts, marker='o', color='tab:red', linestyle='-', linewidth=2, label='Prunable')
        ax.fill_between(layers, counts, alpha=0.1, color='red')
        
        ax.set_title(f"Prunable Capacity @ TopK={k}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Layer Index")
        if i == 0:
            ax.set_ylabel("Count of English-Only Features")
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.suptitle(f"Prunable Feature Analysis ('English-Only' Set) for Task: {task.capitalize()}", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f"prunable_scan_multik_{task}.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    cleanup()
    model = HookedTransformer.from_pretrained(
        MODEL_ID, 
        device="cuda", 
        dtype=torch.bfloat16,
        n_devices=4 
    )
    cleanup()
    TASKS = ['code', 'math']

    for task in TASKS:
        print(f"\n\n=== Experiment: Scanning Prunable Features for {task} ===")
        # # Scan
        # scan_prunable_features(task=task)

        # Scan with multiple K values
        scan_prunable_features_multik(task=task, k_values=[1000, 3000, 5000])