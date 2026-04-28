import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import psutil  
import sys
import glob

def stack_pt_files_incremental(output_dir, file_prefix, save_path, chunk_size=100, verbose=True):
    file_pattern = f"{output_dir}/{file_prefix}/*.pt"
    file_paths = sorted(glob.glob(file_pattern))
    if verbose:
        print(f"Found {len(file_paths)} files to stack in {file_prefix} directory.", flush=True)
    stacked_tensor = None
    for i in range(0, len(file_paths), chunk_size):
        chunk_files = file_paths[i:i + chunk_size]
        chunk_tensors = []
        for idx, file_path in enumerate(chunk_files):
            try:
                if verbose:
                    print(f"[{i + idx + 1}/{len(file_paths)}] Loading file: {file_path}", flush=True)
                tensor = torch.load(file_path)
                chunk_tensors.append(tensor)
                if verbose:
                    print(f"  Loaded tensor shape: {tensor.shape}", flush=True)
            except Exception as e:
                print(f"  ERROR: Failed to load file {file_path} - {e}", file=sys.stderr, flush=True)
                continue
        if not chunk_tensors:
            print(f"No tensors found in chunk {i // chunk_size + 1}. Skipping.", flush=True)
            continue
        chunk_stacked = torch.cat(chunk_tensors, dim=0)
        if stacked_tensor is None:
            stacked_tensor = chunk_stacked
        else:
            stacked_tensor = torch.cat([stacked_tensor, chunk_stacked], dim=0)
        del chunk_tensors, chunk_stacked
        torch.cuda.empty_cache()
        if verbose:
            print(f"Processed chunk {i // chunk_size + 1}. Current stacked shape: {stacked_tensor.shape}", flush=True)
    if stacked_tensor is not None:
        torch.save(stacked_tensor, save_path)
        if verbose:
            print(f"Final stacked tensor saved to {save_path}. Shape: {stacked_tensor.shape}", flush=True)
    else:
        print("No tensors were successfully processed.", flush=True)

output_dir = "/exports/eddie/scratch/s2488128/ESM2/output_800_t12_35m"
save_path_embeddings = f"{output_dir}/stacked_embeddings.pt"
save_path_attentions = f"{output_dir}/stacked_attentions.pt"
stack_pt_files_incremental(output_dir, "embeddings", save_path_embeddings, chunk_size=100, verbose=True)
stack_pt_files_incremental(output_dir, "attention_averages", save_path_attentions, chunk_size=100, verbose=True)