import numpy as np
import pandas as pd
import os

nodes_path = "data/dga_nodes_expanded.csv"
emb_path   = "data/dganet_literature_embeddings_fused.npy"

print("📥 Loading files...")
nodes = pd.read_csv(nodes_path)
emb = np.load(emb_path)

print(f"✅ Nodes: {len(nodes):,} | Embeddings: {emb.shape}")

# --- Basic alignment check ---
aligned = len(nodes) == emb.shape[0]
print(f"\n🧩 1️⃣ Same length? → {aligned}")

if not aligned:
    diff = len(nodes) - emb.shape[0]
    if diff > 0:
        print(f"⚠️ There are {diff:,} more nodes than embeddings.")
    else:
        print(f"⚠️ There are {-diff:,} more embeddings than nodes.")

# --- Check for obvious drift ---
sample_indices = [0, len(nodes)//4, len(nodes)//2, len(nodes)-1]
print("\n🔍 Spot check of node order:")
for idx in sample_indices:
    if idx < len(nodes):
        node_name = nodes.iloc[idx]["Node"]
        emb_norm = np.linalg.norm(emb[idx]) if idx < len(emb) else None
        print(f"{idx:>5} | {node_name:<40} | norm={emb_norm}")

# --- Zero-vector analysis ---
zero_vecs = np.isclose(np.linalg.norm(emb, axis=1), 0).sum()
print(f"\n⚙️ Embeddings with near-zero magnitude: {zero_vecs:,}")

# --- Summary verdict ---
if not aligned or zero_vecs > 0.05 * len(emb):
    print("\n❌ Misalignment or many zero vectors detected — re-fusion recommended.")
else:
    print("\n✅ Embedding alignment looks reasonable.")
