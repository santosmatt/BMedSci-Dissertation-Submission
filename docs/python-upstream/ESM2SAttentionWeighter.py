import os
import torch
import torch.nn.functional as F
import numpy as np
import sys

# Define paths for the stacked files and output directory
output_dir = "/exports/eddie/scratch/s2488128/ESM2/output_800_t12_35m"
stacked_embeddings_path = f"{output_dir}/stacked_embeddings.pt"
stacked_attentions_path = f"{output_dir}/stacked_attentions.pt"

try:
    # Step 1: Load stacked tensors
    print("Loading stacked tensors...", flush=True)
    stacked_embeddings = torch.load(stacked_embeddings_path)  # Shape: [161314, 800, 320]
    print(f"Loaded embeddings: shape = {stacked_embeddings.shape}", flush=True)
    stacked_attentions = torch.load(stacked_attentions_path)  # Shape: [161314, 20, 800]
    print(f"Loaded attentions: shape = {stacked_attentions.shape}", flush=True)

    # Step 2: Average attention weights across heads
    print("Averaging attention weights across heads...", flush=True)
    attention_weights = torch.mean(stacked_attentions, dim=1)  # Shape: [161314, 800]
    print(f"Averaged attention weights: shape = {attention_weights.shape}", flush=True)

    # Step 3: Apply softmax to normalise the attention scores
    print("Normalising attention weights with softmax...", flush=True)
    attention_weights = F.softmax(attention_weights, dim=-1)  # Shape: [161314, 800]
    print("Softmax applied successfully.", flush=True)

    # Step 4: Perform batch matrix multiplication
    print("Computing weighted embeddings with batch matrix multiplication...", flush=True)
    weighted_embeddings = torch.bmm(attention_weights.unsqueeze(1), stacked_embeddings)  # Shape: [161314, 1, 320]
    print(f"Weighted embeddings computed: shape = {weighted_embeddings.shape}", flush=True)

    # Remove the singleton dimension
    weighted_embeddings = weighted_embeddings.squeeze(1)  # Shape: [161314, 320]
    print(f"Final weighted embeddings shape: {weighted_embeddings.shape}", flush=True)

    # Step 5: Save the weighted embeddings
    print(f"Saving weighted embeddings to {output_dir}/weighted_pooled_embeddings.npy...", flush=True)
    np.save(f"{output_dir}/weighted_pooled_embeddings.npy", weighted_embeddings.numpy())
    print(f"Weighted embeddings saved successfully. File size: {os.path.getsize(f'{output_dir}/weighted_pooled_embeddings.npy') / 1e6:.2f} MB", flush=True)

    # Step 6: Compute mean-pooled embeddings (average over sequence dimension)
    print("Computing mean-pooled embeddings...", flush=True)
    mean_pooled_embeddings = torch.mean(stacked_embeddings, dim=1)  # Shape: [161314, 320]
    print(f"Mean-pooled embeddings computed: shape = {mean_pooled_embeddings.shape}", flush=True)

    # Step 7: Save the mean-pooled embeddings
    print(f"Saving mean-pooled embeddings to {output_dir}/pooled_embeddings.npy...", flush=True)
    np.save(f"{output_dir}/pooled_embeddings.npy", mean_pooled_embeddings.numpy())
    print(f"Mean-pooled embeddings saved successfully. File size: {os.path.getsize(f'{output_dir}/pooled_embeddings.npy') / 1e6:.2f} MB", flush=True)
    
    print("Processing complete!", flush=True)

except Exception as e:
    print(f"Error encountered: {e}", file=sys.stderr, flush=True)
    sys.exit(1)