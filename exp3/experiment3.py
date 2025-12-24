import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
import pandas as pd

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "gemma-2-9b"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Sweep Config
# Gemma-2-2B has 26 layers (0-25). We scan all of them.
LAYERS_TO_SCAN = range(0, 42) 
SPARSITY_LEVELS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
     
print(f"Initializing Full Sweep on {DEVICE}...")

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 2. UTILITIES
# ==========================================
def get_task_data(task_name, n_samples=50):
    texts = []
    try:
        if task_name == 'code':
            ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
            for item in ds.take(n_samples):
                texts.append(item['content'])
        elif task_name == 'english':
            ds = load_dataset("wikitext", "wikitext-2-v1", split="train", streaming=True)
            for item in ds.take(n_samples):
                texts.append(item['text'])
    except Exception as e:
        print(f"Error loading {task_name}: {e}")
        return []
    return texts

def measure_perplexity(model, text_list):
    total_loss = 0
    count = 0
    with torch.no_grad():
        for text in text_list:
            tokens = model.to_tokens(text, truncate=True, prepend_bos=True)[:, :512]
            if tokens.shape[1] < 10: continue 
            try:
                loss = model(tokens, return_type="loss")
                if torch.isnan(loss): continue
                total_loss += loss.item()
                count += 1
            except Exception:
                continue
    if count == 0: return float('nan')
    return total_loss / count

# ==========================================
# 3. CORE LOGIC
# ==========================================
def get_differential_features(model, sae, target_texts, contrast_texts, alpha=1.5, top_k=2000):
    total_acts = torch.zeros(sae.cfg.d_sae, device=DEVICE)
    hook_name = f"blocks.{sae.cfg.hook_layer}.hook_resid_post"
    
    def get_mean_acts(texts, accumulator):
        accumulator.zero_()
        tokens = model.to_tokens(texts, truncate=True, prepend_bos=True)[:, :128]
        def hook_fn(activations, hook):
            feature_acts = sae.encode(activations)
            accumulator.add_(feature_acts.sum(dim=(0,1)))
        
        with torch.no_grad():
            model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
        return accumulator.clone() / tokens.numel()

    buff = torch.zeros(sae.cfg.d_sae, device=DEVICE)
    target_acts = get_mean_acts(target_texts, buff)
    contrast_acts = get_mean_acts(contrast_texts, buff)
    
    diff_scores = torch.relu(target_acts - (alpha * contrast_acts))
    return torch.topk(diff_scores, top_k).indices

def compute_attribution_scores(model, sae, layer_idx, target_feature_indices):
    W_out = model.blocks[layer_idx].mlp.W_out.data
    W_out_norm = torch.nn.functional.normalize(W_out, dim=1)

    W_dec_targets = sae.W_dec.data[target_feature_indices]
    W_dec_norm = torch.nn.functional.normalize(W_dec_targets, dim=1)

    neuron_scores = torch.zeros(W_out.shape[0], device=DEVICE, dtype=W_out.dtype)
    chunk_size = 1000
    
    for i in range(0, len(target_feature_indices), chunk_size):
        chunk_dec = W_dec_norm[i : i+chunk_size].to(dtype=W_out.dtype)
        sim_matrix = torch.matmul(W_out_norm, chunk_dec.T)
        neuron_scores += torch.relu(sim_matrix).sum(dim=1)
        
    return neuron_scores

# ==========================================
# 4. PRUNING HELPERS
# ==========================================
def get_layer_weights(model, layer_idx):
    return {
        "W_in": model.blocks[layer_idx].mlp.W_in.data.clone().cpu(),
        "W_out": model.blocks[layer_idx].mlp.W_out.data.clone().cpu(),
        "b_in": model.blocks[layer_idx].mlp.b_in.data.clone().cpu() if model.blocks[layer_idx].mlp.b_in is not None else None,
    }

def restore_layer_weights(model, layer_idx, backup):
    with torch.no_grad():
        model.blocks[layer_idx].mlp.W_in.data = backup["W_in"].to(DEVICE)
        model.blocks[layer_idx].mlp.W_out.data = backup["W_out"].to(DEVICE)
        if backup["b_in"] is not None:
            model.blocks[layer_idx].mlp.b_in.data = backup["b_in"].to(DEVICE)

def apply_pruning_mask(model, mask, layer_idx):
    mask = mask.to(DEVICE)
    with torch.no_grad():
        model.blocks[layer_idx].mlp.W_in.data[:, ~mask] = 0
        model.blocks[layer_idx].mlp.W_out.data[~mask, :] = 0
        if model.blocks[layer_idx].mlp.b_in is not None:
             model.blocks[layer_idx].mlp.b_in.data[~mask] = 0

# ==========================================
# 5. MAIN SWEEP LOOP
# ==========================================

if __name__ == "__main__":
    
    # 1. Load Model
    print(f"Loading Model {MODEL_ID}...")
    model = HookedTransformer.from_pretrained(
        MODEL_ID, device=DEVICE, dtype=torch.bfloat16, default_prepend_bos=True
    )
    cleanup()
    
    # Load Data once
    target_texts = get_task_data('code', n_samples=50)
    contrast_texts = get_task_data('english', n_samples=50)
    
    # Storage for results: [Layer, Sparsity, Mag_Loss, SAE_Loss]
    results_log = []

    print(f"\nStarting Sweep across {len(LAYERS_TO_SCAN)} layers...")
    print(f"Sparsity Levels: {SPARSITY_LEVELS}")

    for layer_idx in LAYERS_TO_SCAN:
        print(f"\n=== Processing Layer {layer_idx} ===")
        
        # A. Load SAE for this layer
        sae_id = f"layer_{layer_idx}/width_16k/canonical"
        try:
            sae, _, _ = SAE.from_pretrained(SAE_RELEASE, sae_id, device=DEVICE)
        except Exception as e:
            print(f"Skipping Layer {layer_idx} (No SAE found): {e}")
            continue

        # B. Compute Attribution (The "Map")
        try:
            target_indices = get_differential_features(model, sae, target_texts, contrast_texts)
            neuron_scores = compute_attribution_scores(model, sae, layer_idx, target_indices)
        except Exception as e:
            print(f"Error computing attribution for Layer {layer_idx}: {e}")
            del sae
            cleanup()
            continue
            
        del sae
        cleanup()

        # C. Run Sparsity Sweep
        original_weights = get_layer_weights(model, layer_idx)
        mag_scores = model.blocks[layer_idx].mlp.W_out.data.norm(dim=1)
        
        for sparsity in SPARSITY_LEVELS:
            k = int(len(mag_scores) * (1 - sparsity))
            
            # 1. Magnitude Pruning
            # Reset
            restore_layer_weights(model, layer_idx, original_weights)
            _, keep_mag = torch.topk(mag_scores, k)
            mask_mag = torch.zeros_like(mag_scores, dtype=torch.bool)
            mask_mag[keep_mag] = True
            apply_pruning_mask(model, mask_mag, layer_idx)
            
            loss_mag = measure_perplexity(model, target_texts[:20])

            # 2. SAE Pruning
            # Reset
            restore_layer_weights(model, layer_idx, original_weights)
            _, keep_sae = torch.topk(neuron_scores, k)
            mask_sae = torch.zeros_like(neuron_scores, dtype=torch.bool)
            mask_sae[keep_sae] = True
            apply_pruning_mask(model, mask_sae, layer_idx)
            
            loss_sae = measure_perplexity(model, target_texts[:20])

            # Log result
            diff = loss_mag - loss_sae # Positive = SAE Wins
            print(f"   Sparsity {sparsity:.2f} | Mag: {loss_mag:.4f} | SAE: {loss_sae:.4f} | Diff: {diff:+.4f}")
            results_log.append({
                "Layer": layer_idx,
                "Sparsity": sparsity,
                "Mag_Loss": loss_mag,
                "SAE_Loss": loss_sae,
                "Improvement": diff
            })

        # D. Final Reset for this layer before moving to next
        restore_layer_weights(model, layer_idx, original_weights)
        cleanup()

    # ==========================================
    # 6. VISUALIZATION (HEATMAP)
    # ==========================================
    if results_log:
        df = pd.DataFrame(results_log)
        
        # Save raw data
        df.to_csv("sae_pruning_full_sweep.csv", index=False)
        print("\nSweep Complete. Data saved to 'sae_pruning_full_sweep.csv'.")

        # Create Heatmap Matrix
        # Pivot: Index=Layer, Columns=Sparsity, Values=Improvement
        pivot_table = df.pivot(index="Layer", columns="Sparsity", values="Improvement")
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap="RdYlGn", center=0, fmt=".3f")
        plt.title("SAE-Guided vs Magnitude Pruning (Green = SAE Wins)")
        plt.ylabel("Layer Index")
        plt.xlabel("Sparsity Ratio")
        plt.savefig("pruning_heatmap.png")
        print("Heatmap saved to 'pruning_heatmap.png'")
    else:
        print("No results to plot.")