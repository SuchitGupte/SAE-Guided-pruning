import torch
import gc
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
import pandas as pd


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
        # activations: [batch, seq, d_model] on layer's device 
        acts = activations.to(sae.device)
        
        feature_acts = sae.encode(acts)
        # Sum over batch and sequence length
        total_acts.add_(feature_acts.sum(dim=(0,1)))

    # Tokenize and run
    tokens = model.to_tokens(text_list, truncate=True)[:, :128]

    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    
    return torch.topk(total_acts, top_k).indices

def run_differential_sweep(model, task_texts, baseline_texts, layers_to_check=range(0, 26, 1)):
    """
    For each layer, identifies features unique to the task 
    (Task Top-K minus Baseline Top-K).
    """
    results = {}
    
    for layer in layers_to_check:
        print(f"\n--- Analyzing Layer {layer} ---")
        sae = None
        try:
            # Load SAE for this layer
            release = f"gemma-scope-2b-pt-{OUT_TYPE}-canonical" 
            sae_id = f"layer_{layer}/width_16k/canonical" 

            # release = f"gemma-scope-9b-pt-{OUT_TYPE}-canonical"
            # sae_id = f"layer_{layer}/width_16k/canonical"

            
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            # Ensure SAE is on the correct analysis device
            sae.to(DEVICE)
            
            # Get Feature Sets

            task_indices = set(get_active_features(model, sae, task_texts, top_k=100).tolist())
            baseline_indices = set(get_active_features(model, sae, baseline_texts, top_k=100).tolist())

            # The Core Metric: How many task features are NOT in the baseline?
            unique_task_features = task_indices - baseline_indices
            unique_count = len(unique_task_features)
            
            # Intersection for context
            shared_count = len(task_indices.intersection(baseline_indices))

            print(f"  Shared (Generic): {shared_count}")
            print(f"  Unique (Task-Specific): {unique_count}")
            
            results[layer] = unique_count
            
            del sae
            cleanup()
            
        except Exception as e:
            print(f"Could not load/process Layer {layer}: {e}")
            
    return results

def get_feature_sums(model, sae, text_list):
    """
    Returns the tensor of total activations for ALL features.
    We return the raw sums so we can slice Top-K later without re-running the model.
    """
    total_acts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    hook_name = f"blocks.{sae.cfg.hook_layer}.hook_resid_post"

    def hook_fn(activations, hook):
        acts = activations.to(sae.device)
        feature_acts = sae.encode(acts)
        total_acts.add_(feature_acts.sum(dim=(0,1)))

    tokens = model.to_tokens(text_list, truncate=True)[:, :128]
    with torch.no_grad():
        model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    
    return total_acts

def run_differential_sweep_multi_k(model, task_name, task_texts, baseline_texts, 
                                   layers_to_check, k_values=[50, 100, 200]):
    """
    Collects Unique/Shared stats for multiple Top-K thresholds efficiently.
    """
    data_records = []
    
    for layer in layers_to_check:
        print(f"--- Processing {task_name} | Layer {layer} ---")
        try:
            # 1. Load SAE
            release = f"gemma-scope-2b-pt-{OUT_TYPE}-canonical" 
            sae_id = f"layer_{layer}/width_16k/canonical"

            # release = f"gemma-scope-9b-pt-{OUT_TYPE}-canonical"
            # sae_id = f"layer_{layer}/width_16k/canonical" 
            
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            sae.to(DEVICE)
            
            # 2. Get All Activations (Computationally expensive part done once)
            task_acts = get_feature_sums(model, sae, task_texts)
            base_acts = get_feature_sums(model, sae, baseline_texts)
            
            # 3. Analyze for different K values (Cheap operation)
            for k in k_values:
                # Get indices of top K
                task_indices = set(torch.topk(task_acts, k).indices.tolist())
                base_indices = set(torch.topk(base_acts, k).indices.tolist())
                
                # Metrics
                unique = len(task_indices - base_indices)
                shared = len(task_indices.intersection(base_indices))
                
                data_records.append({
                    "Task": task_name,
                    "Layer": layer,
                    "TopK": k,
                    "Metric": "Unique",
                    "Value": unique
                })
                data_records.append({
                    "Task": task_name,
                    "Layer": layer,
                    "TopK": k,
                    "Metric": "Shared",
                    "Value": shared
                })

            del sae
            cleanup()
            
        except Exception as e:
            print(f"Error Layer {layer}: {e}")
            
    return data_records

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

if __name__=="__main__":
    cleanup()
    model = HookedTransformer.from_pretrained(
        MODEL_ID, 
        device="cuda", 
        dtype=torch.bfloat16,
        n_devices=4 
    )
    cleanup()
    TASKS = ['code', 'math']
    
    # Get Baseline Once
    print("Loading Baseline Data (English)...")
    baseline_texts = get_task_data('english', n_samples=100)

    records = []
    for task in TASKS:
        task_texts = get_task_data(task)
        task_data = run_differential_sweep_multi_k(model, task, task_texts, baseline_texts, range(0, 26, 1), k_values=[50, 100, 500, 1000, 2000, 3000, 5000])
        # task_data = run_differential_sweep_multi_k(model, task, task_texts, baseline_texts, range(0, 42, 1), k_values=[50, 100, 500, 1000, 2000, 3000, 5000])
        records.extend(task_data)
        cleanup()

    df_results = pd.DataFrame(records)
    df_results.to_csv("sae_analysis_results.csv", index=False)
    
    
    
