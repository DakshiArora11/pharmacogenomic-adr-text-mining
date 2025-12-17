import pandas as pd
import numpy as np
import os

# --- Config ---
nodes_path = "data/dga_nodes_expanded.csv"
graph_emb_path = "data/dganet_literature_embeddings_new.npy"
literature_emb_path = "data/literature_features.npy"   # precomputed BioBERT embeddings
out_path = "data/dganet_literature_embeddings_fused.npy"

print("📥 Loading data...")

nodes = pd.read_csv(nodes_path)
graph_emb = np.load(graph_emb_path)
print(f"✅ Graph embeddings: {graph_emb.shape}")

# --- Try to load literature features ---
if os.path.exists(literature_emb_path):
    lit_emb = np.load(literature_emb_path)
    print(f"✅ Literature embeddings: {lit_emb.shape}")
else:
    print("⚠️ No literature_features.npy found — creating zero placeholders.")
    lit_emb = np.zeros((len(nodes), 768))

# --- Sanity check ---
assert graph_emb.shape[0] == len(nodes), "❌ Graph embedding count must match nodes"
if lit_emb.shape[0] != len(nodes):
    print("⚠️ Mismatched literature embedding count; re-aligning/truncating.")
    min_n = min(lit_emb.shape[0], len(nodes))
    lit_emb = lit_emb[:min_n]
    graph_emb = graph_emb[:min_n]
    nodes = nodes.iloc[:min_n]

# --- Fusion strategy: concatenate ---
fused = np.concatenate([graph_emb, lit_emb], axis=1)
print(f"🧬 Fused embedding shape: {fused.shape}")

# --- Save fused embeddings ---
np.save(out_path, fused)
print(f"💾 Saved fused embeddings → {out_path}")

# --- Optional: preview example ---
print("✅ Example vector snippet:", fused[0][:10])
