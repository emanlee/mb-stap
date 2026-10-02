#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tf_predictions_by_segment.py
Restore actual TF names from zdata_tf*.npy files in Nx/ directory,
combine with predictions (y_predict + gene_index) from predict_dir,
and save per-TF CSVs for each segment.
Modify the paths at the top to run.
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# ======== Paths (modify these) ========
predict_dir = "/home/yanke/MB/predict/a1_pdx322_predict_results_no_y_den1"
nx_dir = "/home/yanke/MB/predict/Nx"
tf_target_path = "/home/yanke/MB/predict/TF-target.txt"
out_dir = "/home/yanke/MB/predict/predictions_by_tf"
y_file = "/home/yanke/MB/predict/a1_pdx322_y_predict_den1.npy"
index_file = "/home/yanke/MB/predict/a1_pdx322_gene_index_den1.txt"
# =====================================

def safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

def find_zdata_files(nx_dir):
    """Return (index, filepath) list and sorted filepath list for zdata files"""
    pattern = os.path.join(nx_dir, "zdata_tf*.npy")
    files = glob.glob(pattern)
    pairs = []
    for f in files:
        m = re.search(r'zdata_tf(\d+)\.npy$', os.path.basename(f))
        idx = int(m.group(1)) if m else None
        pairs.append((idx, f))
    pairs.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1]))
    sorted_files = [p[1] for p in pairs]
    sorted_indices = [p[0] for p in pairs]
    return sorted_indices, sorted_files

def extract_tf_name_from_zdata(zfile):
    """Extract TF name from zdata file; return None if parsing fails"""
    try:
        arr = np.load(zfile, allow_pickle=True)
    except Exception as e:
        print(f"⚠️ Cannot read {zfile}: {e}")
        return None
    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return None

    first = None
    try:
        for item in arr:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                first = s
                break
    except Exception:
        first = str(arr[0]).strip() if arr.size > 0 else None

    if not first:
        return None

    if '\t' in first:
        tf = first.split('\t')[0].strip()
        if tf:
            return tf
    parts = re.split(r'\s+', first)
    return parts[0].strip() if parts and parts[0] else None

def main():
    print("🚀 Start: Restore TF names from zdata files and save predictions by TF.")
    
    # Find zdata files
    z_indices, z_files = find_zdata_files(nx_dir)
    if not z_files:
        raise FileNotFoundError(f"No zdata_tf*.npy files found in {nx_dir}.")
    print(f"🔹 Found {len(z_files)} zdata files in {nx_dir} (sorted by index/name).")

    # Extract TF names from zdata files
    tf_names = []
    for f in z_files:
        tfname = extract_tf_name_from_zdata(f)
        if tfname is None:
            print(f"⚠️ Cannot parse TF name from {os.path.basename(f)}, will fallback to TF-target.txt.")
            tf_names.append(None)
        else:
            tf_names.append(tfname)
    print(f"🔹 Example first 5 parsed TF names: {tf_names[:5]}")

    # Load predictions
    y_pred_path = os.path.join(predict_dir, y_file)
    gene_index_path = os.path.join(predict_dir, index_file)
    if not os.path.exists(y_pred_path):
        raise FileNotFoundError(f"Prediction file not found: {y_pred_path}")
    if not os.path.exists(gene_index_path):
        raise FileNotFoundError(f"Gene index file not found: {gene_index_path}")

    y_pred = np.load(y_pred_path)
    gene_index = np.loadtxt(gene_index_path, dtype=int)
    print(f"✅ y_pred.shape = {y_pred.shape}")
    print(f"✅ gene_index = {gene_index.tolist()}")

    if y_pred.ndim != 2 or y_pred.shape[1] < 2:
        raise ValueError("y_pred shape is invalid; expected (N, num_classes).")

    num_classes = y_pred.shape[1]
    print(f"🔹 Detected num_classes = {num_classes}")

    # Load TF-target file
    tf_target = pd.read_csv(tf_target_path, sep="\t", header=None, names=["TF", "Target"], dtype=str)
    print(f"✅ Loaded TF-target.txt with {len(tf_target)} records.")

    os.makedirs(out_dir, exist_ok=True)

    n_segments = len(gene_index) - 1
    if n_segments == 0:
        raise ValueError("gene_index seems to contain only 0 or 1 element.")

    if n_segments != len(z_files):
        print(f"⚠️ gene_index segments = {n_segments}, zdata files = {len(z_files)}. Will iterate segments using TF names where possible.")

    offset = 0
    for seg in range(n_segments):
        start = int(gene_index[seg])
        end = int(gene_index[seg + 1])
        seg_len = end - start
        if seg_len <= 0:
            print(f"⚠️ Segment {seg} length = {seg_len}, skip.")
            continue

        if offset >= y_pred.shape[0]:
            print(f"🔻 No more predictions available (offset={offset} >= {y_pred.shape[0]}). Stop.")
            break
        use_len = min(seg_len, y_pred.shape[0] - offset)
        sub_pred = y_pred[offset: offset + use_len]

        # Determine TF name
        tf_name = tf_names[seg] if seg < len(tf_names) and tf_names[seg] else None
        if tf_name is None:
            try:
                tf_name = tf_target.iloc[start]["TF"]
                print(f"ℹ️ Segment {seg} fallback TF name: {tf_name} (from TF-target row {start})")
            except Exception as e:
                tf_name = f"TF_seg{seg}"
                print(f"⚠️ Segment {seg} fallback placeholder TF name {tf_name}. Error: {e}")

        # Slice targets from TF-target
        sub_targets = tf_target.iloc[start: start + use_len].reset_index(drop=True)
        if len(sub_targets) != use_len:
            print(f"⚠️ Segment {seg} slice length {len(sub_targets)} != use_len {use_len}. Save actual rows.")

        # Convert names to lowercase
        tf_name_lower = str(tf_name).lower()
        targets_lower = sub_targets["Target"].astype(str).str.lower().tolist()

        data = {"TF": [tf_name_lower] * len(sub_targets), "Target": targets_lower}
        for c in range(num_classes):
            data[f"prob_class{c}"] = sub_pred[:, c] if c < sub_pred.shape[1] else np.nan
        data["predicted_class"] = np.argmax(sub_pred, axis=1).astype(int)

        df = pd.DataFrame(data)
        out_name = safe_filename(f"{tf_name}_predictions_seg{seg}.csv")
        out_path = os.path.join(out_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"Saved segment {seg}: TF={tf_name}, rows={len(df)} → {out_path}")

        offset += use_len

    print("All segments processed.")

if __name__ == "__main__":
    main()