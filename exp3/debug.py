import torch
from transformer_lens import HookedTransformer

# 1. Load on ONE device only
print("Loading Model on Single GPU...")
model = HookedTransformer.from_pretrained(
    "gemma-2-2b", 
    device="cuda:0",  # Force single device
    dtype=torch.bfloat16,
    default_prepend_bos=True
)

# 2. Simple Test
text = "The capital of France is"
tokens = model.to_tokens(text)
logits = model(tokens)

# 3. Check Prediction
# Get the predicted token for the last position
predicted_token = logits[0, -1].argmax().item()
predicted_word = model.to_string(predicted_token)

# 4. Check Loss
loss = model(tokens, return_type="loss")

print(f"\nTest Sentence: '{text}'")
print(f"Predicted next word: '{predicted_word}' (Expected: ' Paris')")
print(f"Loss: {loss.item():.4f} (Expected: < 4.0)")

if loss.item() > 10.0:
    print("\nCRITICAL FAILURE: Model is random guessing.")
    print("Please run: pip install -U transformer_lens transformers accelerate")
else:
    print("\nSUCCESS: Model is working.")