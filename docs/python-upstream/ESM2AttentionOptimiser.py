import os
import numpy as np
from scipy.optimize import minimize

# Define loss functions
def total_loss(params, vectors, target_loss=0):
    alpha, beta = params
    total_loss_value = 0
    for vector in vectors:  # Vectors are already transposed (optimising over columns)
        vector_sum = np.sum(vector)
        vector_std = np.std(vector)
        total_loss_value += -(alpha * vector_sum - beta * vector_std)
    penalty_loss = (total_loss_value - target_loss) ** 2
    return penalty_loss

def refined_loss_function(vector, alpha, beta):
    vector_sum = np.sum(vector)
    vector_std = np.std(vector)
    return -(alpha * vector_sum - beta * vector_std)

# Define optimisation function
def optimiseAttention(vectors, num_returned, output_dir, initial_params=[10, 1], target_loss=0):
    print("Starting optimisation over columns...", flush=True)
    result = minimize(total_loss, initial_params, args=(vectors, target_loss))
    optimal_alpha, optimal_beta = result.x
    print(f"Optimal alpha: {optimal_alpha}, Optimal beta: {optimal_beta}", flush=True)

    # Compute refined losses for all columns
    losses = [refined_loss_function(vector, optimal_alpha, optimal_beta) for vector in vectors]
    top_indices = np.argsort(losses)[:num_returned]  # Top columns

    # Save top indices
    results_path = os.path.join(output_dir, "top_attention_column_indices.npy")
    np.save(results_path, top_indices)
    print(f"Top column indices saved to {results_path}", flush=True)

    print(f"\nTop {num_returned} columns with the lowest loss:", flush=True)
    for i in top_indices:
        print(f"Column {i} with loss {losses[i]}", flush=True)
    
    return top_indices

# Main script
if __name__ == "__main__":
    # Paths and parameters
    output_dir = "/exports/eddie/scratch/s2488128/ESM2/output_800_t12_35m"
    weighted_embeddings_path = f"{output_dir}/weighted_pooled_embeddings.npy"
    non_weighted_embeddings_path = f"{output_dir}/pooled_embeddings.npy"
    num_returned = 10  # Number of top columns to return
    initial_params = [10, 1]  # Initial guess for alpha and beta

    # Load weighted pooled embeddings
    print("Loading weighted pooled embeddings...", flush=True)
    weighted_embeddings = np.load(weighted_embeddings_path).transpose()  # Shape: [embedding_dim, num_sequences]
    print(f"Loaded and transposed weighted embeddings: shape = {weighted_embeddings.shape}", flush=True)

    # Load unweighted pooled embeddings
    print("Loading unweighted pooled embeddings...", flush=True)
    unweighted_embeddings = np.load(non_weighted_embeddings_path)  # Shape: [num_sequences, embedding_dim]
    print(f"Loaded unweighted embeddings: shape = {unweighted_embeddings.shape}", flush=True)

    # Run optimisation on weighted embeddings
    print("Running optimisation on transposed weighted embeddings (columns)...", flush=True)
    top_indices = optimiseAttention(weighted_embeddings, num_returned, output_dir, initial_params, target_loss=0)

    # Select top columns from the unweighted embeddings
    print("Selecting top embedding columns from unweighted embeddings...", flush=True)
    selected_unweighted_embeddings = unweighted_embeddings[:, top_indices]  # Shape: [num_sequences, num_returned]

    # Save the selected unweighted embeddings
    selected_embeddings_path = f"{output_dir}/selected_top_unweighted_embeddings.npy"
    np.save(selected_embeddings_path, selected_unweighted_embeddings)
    print(f"Selected unweighted embeddings saved to {selected_embeddings_path}", flush=True)