import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import psutil  
import sys
import glob

DATA_DIR = "/exports/eddie/scratch/s2488128/ESM2/data"
OUTPUT_DIR = "/exports/eddie/scratch/s2488128/ESM2/output_800_t12_35m"
SEQUENCE_FILE = '/exports/eddie/scratch/s2488128/ESM2/data/sequences.txt'
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
BATCH_SIZE = 500
MAX_LENGTH = 800

def log_memory_usage(batch_index):
    """Logs memory usage for debugging and monitoring."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_in_gb = memory_info.rss / 1e9  # Resident Set Size (GB)
    vms_in_gb = memory_info.vms / 1e9  # Virtual Memory Size (GB)
    system_memory = psutil.virtual_memory()
    total_memory_gb = system_memory.total / 1e9
    used_memory_gb = system_memory.used / 1e9
    available_memory_gb = system_memory.available / 1e9
    proportion_used = (rss_in_gb / total_memory_gb) * 100
    print(
        f"Batch {batch_index}: "
        f"RSS = {rss_in_gb:.2f} GB, VMS = {vms_in_gb:.2f} GB, "
        f"Available = {available_memory_gb:.2f} GB, Total = {total_memory_gb:.2f} GB, "
        f"Proportion Used = {proportion_used:.2f}%",
        file=sys.stderr
    )

def log_file_size(file_path, batch_index):
    """Logs the size of a saved file."""
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / 1e9  # Convert to GB
        print(f"Batch {batch_index}: File saved: {file_path} ({file_size:.2f} GB)", file=sys.stderr)

def generateESM2(model_name, sequence_data, batch_size, max_length, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cpu')  # Force CPU
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    os.makedirs(f"{output_dir}/attention_averages", exist_ok=True)
    os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
    pro_seq = sequence_data['sequence'].tolist()
    for i in tqdm(range(0, len(pro_seq), batch_size), desc="Processing Sequences"):
        attention_avg_path = f"{output_dir}/attention_averages/attention_avg_batch_{i}.pt"
        embeddings_path = f"{output_dir}/embeddings/embeddings_batch_{i}.pt"
        if os.path.exists(attention_avg_path) and os.path.exists(embeddings_path):
            print(f"Batch {i} already processed. Skipping.")
            continue        
        batch_seqs = pro_seq[i:i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            # Accumulate attention averages across layers
            num_layers = len(outputs.attentions)
            batch_attention_mean = None
            for layer_attention in outputs.attentions:  # Iterate over layers
                # Average across heads (dim=2) for the current layer
                layer_attention_mean = layer_attention.mean(dim=2)  # Shape: (batch_size, seq_len, seq_len)
                if batch_attention_mean is None:
                    batch_attention_mean = layer_attention_mean
                else:
                    batch_attention_mean += layer_attention_mean
            batch_attention_mean /= num_layers  # Shape: (batch_size, seq_len, seq_len)
            attention_avg_path = f"{output_dir}/attention_averages/attention_avg_batch_{i}.pt"
            torch.save(batch_attention_mean, attention_avg_path)
            print(f"Saved attention averages for batch {i} to {attention_avg_path}")
            batch_embds = outputs.last_hidden_state.cpu()
            embeddings_path = f"{output_dir}/embeddings/embeddings_batch_{i}.pt"
            torch.save(batch_embds, embeddings_path)
            print(f"Saved embeddings for batch {i} to {embeddings_path}")
        log_memory_usage(i)
        del input_ids, attention_mask, outputs, batch_embds, batch_attention_mean
        torch.cuda.empty_cache()

if __name__ == "__main__":
    sequence_data = pd.read_csv(SEQUENCE_FILE, sep="\t")
    generateESM2(MODEL_NAME, sequence_data, BATCH_SIZE, MAX_LENGTH, OUTPUT_DIR)
    print("Processing complete!")