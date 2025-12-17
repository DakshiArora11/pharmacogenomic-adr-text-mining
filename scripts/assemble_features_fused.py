import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

# --- CLI Arguments ---
p = argparse.ArgumentParser()
p.add_argument("--triples_csv", default="data/triples_normalized.csv")
p.add_argument("--emb_file", default="data/dganet_literature_embeddings_fused.npy")
p.add_argument("--nodes_csv", default="data/dga_nodes_expanded.csv")
p.add_argument("--out", default="output/features_fused.csv")
args = p.parse_args()

print("📥 Loading data...")
triples = pd.read_csv(args.triples_csv)
nodes = pd.read_csv(args.nodes_csv)
embeddings = np.load(args.emb_file)

print(f"✅ Loaded {len(triples):,} triples, {len(nodes):,} nodes, embeddings shape: {embeddings.shape}")

# --- Map nodes to indices ---
node_map = {n: i for i, n in enumerate(nodes["Node"])}

missing_count = 0
dim = embeddings.shape[1]

# --- Helper for safe vector retrieval ---
def get_vec(node_name):
    global missing_count
    idx = node_map.get(node_name)
    if idx is None or idx >= len(embeddings):
        missing_count += 1
        return np.zeros(dim)
    return embeddings[idx]

def clean(s): 
    return str(s).strip().replace(" ", "_")

# --- Build feature table ---
rows = []
for _, r in tqdm(triples.iterrows(), total=len(triples), desc="🔧 Assembling features"):
    d = clean(r.get("Drug", ""))
    g = clean(r.get("Gene", ""))
    a = clean(r.get("ADR", ""))
    mesh = r.get("MeSH_ID", "")

    drug_key = f"DRUG_{d.lower()}"
    gene_key = f"GENE_{g.upper()}"
    adr_key  = f"ADR_{a}"

    dvec = get_vec(drug_key)
    gvec = get_vec(gene_key)
    avec = get_vec(adr_key)

    feature_vec = np.concatenate([dvec, gvec, avec])
    rows.append([d, g, a, mesh] + feature_vec.tolist())

# --- Create DataFrame ---
header = ["drug", "gene", "adr", "mesh_id"] + [f"f{i}" for i in range(len(rows[0]) - 4)]
out_df = pd.DataFrame(rows, columns=header)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
out_df.to_csv(args.out, index=False)

print(f"💾 Saved fused feature table → {args.out}")
print(f"📊 Shape: {out_df.shape}")
print(f"⚠️ Missing embeddings for {missing_count:,} nodes (filled with zeros)")
print("✅ Example rows:")
print(out_df.head(3))
