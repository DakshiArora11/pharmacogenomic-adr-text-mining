import numpy as np
import pandas as pd
import os
import argparse

p = argparse.ArgumentParser()
p.add_argument("--nodes_csv", default="data/dga_nodes_expanded.csv")
p.add_argument("--graph_emb", default="data/dganet_literature_embeddings_new.npy")
p.add_argument("--lit_emb", default="data/literature_embeddings.npy")  # adjust if needed
p.add_argument("--out", default="data/dganet_literature_embeddings_fused_aligned.npy")
args = p.parse_args()

print("📥 Loading data...")
nodes = pd.read_csv(args.nodes_csv)
graph_emb = np.load(args.graph_emb)
lit_emb = np.load(args.lit_emb)

from sklearn.preprocessing import StandardScaler

# Normalize embeddings before fusion
scaler_g = StandardScaler()
scaler_l = StandardScaler()
graph_emb = scaler_g.fit_transform(graph_emb)
lit_emb = scaler_l.fit_transform(lit_emb)


print(f"✅ Graph embeddings: {graph_emb.shape}")
print(f"✅ Literature embeddings: {lit_emb.shape}")
print(f"✅ Node count: {len(nodes):,}")

# --- Build mapping between node names and available embeddings ---
# Assume literature embeddings file has corresponding node list if available
lit_nodes_file = "data/literature_nodes.csv"
if os.path.exists(lit_nodes_file):
    lit_nodes = pd.read_csv(lit_nodes_file)
    lit_map = {n: i for i, n in enumerate(lit_nodes["Node"])}
    print(f"✅ Loaded literature node list: {len(lit_nodes):,}")
else:
    print("⚠️ No literature_nodes.csv found — assuming 1:1 order with first 7892 nodes.")
    lit_map = {n: i for i, n in enumerate(nodes["Node"][:len(lit_emb)])}

# --- Fuse per node ---
fused = []
dim_g, dim_l = graph_emb.shape[1], lit_emb.shape[1]

for n in nodes["Node"]:
    g_idx = nodes.index[nodes["Node"] == n][0]
    g_vec = graph_emb[g_idx] if g_idx < len(graph_emb) else np.zeros(dim_g)
    l_idx = lit_map.get(n)
    l_vec = lit_emb[l_idx] if l_idx is not None and l_idx < len(lit_emb) else np.zeros(dim_l)
    fused_vec = np.concatenate([g_vec, l_vec])
    fused.append(fused_vec)

fused = np.array(fused)
np.save(args.out, fused)

print(f"💾 Saved aligned fused embeddings → {args.out}")
print("🧩 Final shape:", fused.shape)
print("✅ Example vector snippet:", fused[0][:10])
