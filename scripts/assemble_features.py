import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -------------------- ARGUMENTS --------------------
p = argparse.ArgumentParser(description="Assemble fused features for DGANet-Literature model")
p.add_argument("--triples_csv", default="data/triples_normalized.csv", help="Input triples with Drug/Gene/ADR/MeSH_ID")
p.add_argument("--emb_file", default="data/dganet_literature_embeddings.npy", help="Node embeddings (numpy array)")
p.add_argument("--nodes_csv", default="data/dga_nodes.csv", help="Node name mapping")
p.add_argument("--literature_feats", default="data/literature_features.npy", help="Optional literature features")
p.add_argument("--out", default="output/features_fused.csv", help="Output feature CSV")
p.add_argument("--neg_ratio", type=int, default=1, help="Negative:positive sampling ratio")
args = p.parse_args()

# -------------------- LOAD DATA --------------------
print("📥 Loading data...")

triples = pd.read_csv(args.triples_csv)
nodes = pd.read_csv(args.nodes_csv)
embeddings = np.load(args.emb_file)
node_map = dict(zip(nodes["Node"], range(len(nodes))))

# Optional: literature features (if file exists)
literature_feats = None
if os.path.exists(args.literature_feats):
    literature_feats = np.load(args.literature_feats)
    print(f"✅ Loaded literature features with shape {literature_feats.shape}")
else:
    print("⚠️ No literature_features.npy found — skipping literature pooling")

print(f"✅ Loaded {len(triples)} triples and {len(nodes)} nodes")

# -------------------- CLEAN + NORMALIZE --------------------
triples.columns = [c.strip().lower() for c in triples.columns]
triples = triples.rename(columns={"drug": "drugs", "gene": "genes", "adr": "adrs", "mesh_id": "mesh_id"})
triples = triples.dropna(subset=["drugs", "adrs"])
triples["drugs"] = triples["drugs"].astype(str)
triples["genes"] = triples["genes"].fillna("").astype(str)
triples["adrs"] = triples["adrs"].astype(str)
triples["mesh_id"] = triples["mesh_id"].fillna("").astype(str)
triples["label"] = 1

# -------------------- NEGATIVE SAMPLING --------------------
unique_drugs = triples["drugs"].unique().tolist()
unique_adrs = triples["adrs"].unique().tolist()

neg_rows = []
num_neg = len(triples) * args.neg_ratio
print(f"🧪 Sampling {num_neg} negative triples...")

for _ in tqdm(range(num_neg)):
    drug_choice = np.random.choice(unique_drugs)
    adr_choice = np.random.choice(unique_adrs)
    # Skip if positive already exists
    if ((triples["drugs"] == drug_choice) & (triples["adrs"] == adr_choice)).any():
        continue
    neg_rows.append({
        "drugs": drug_choice,
        "genes": "",
        "adrs": adr_choice,
        "mesh_id": "",
        "label": 0
    })

neg_df = pd.DataFrame(neg_rows)
data_all = pd.concat([triples, neg_df], ignore_index=True)
print(f"✅ Final dataset size: {len(data_all)} rows")

# -------------------- NORMALIZED LOOKUPS --------------------
def normalize_node_name(name: str):
    return str(name).strip().replace(" ", "_").upper()

norm_node_map = {normalize_node_name(k): v for k, v in node_map.items()}

def get_vec(node_name):
    key = normalize_node_name(node_name)
    if key in norm_node_map:
        idx = norm_node_map[key]
        return embeddings[idx]
    return np.zeros(embeddings.shape[1])

def safe_cosine(v1, v2):
    if np.all(v1 == 0) or np.all(v2 == 0):
        return 0.0
    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]

# -------------------- FEATURE ASSEMBLY --------------------
rows = []
dim = embeddings.shape[1]
print(f"⚙️ Building feature vectors (dim={dim})")

def normalize_drug_name(name: str):
    """Normalize drug names to match node pattern."""
    return name.strip().replace(" ", "_").lower()

def normalize_mesh(mesh: str):
    """Normalize MeSH IDs for ADR lookup."""
    return mesh.strip().upper().replace("MESH:", "")

def get_vec_by_key(node_key: str):
    """Get embedding vector from normalized node name."""
    if not node_key:
        return np.zeros(dim)
    key = node_key.strip()
    if key in node_map:
        idx = node_map[key]
        return embeddings[idx]
    return np.zeros(dim)

for _, row in tqdm(data_all.iterrows(), total=len(data_all)):
    d = normalize_drug_name(row["drugs"])
    g = row["genes"].strip().replace(" ", "_") if pd.notna(row["genes"]) else ""
    a = row["adrs"].strip()
    mesh_id = normalize_mesh(row["mesh_id"]) if pd.notna(row["mesh_id"]) else ""

    # Build proper node keys
    d_key = f"DRUG_{d}"
    g_key = f"GENE_{g}" if g else None
    a_key = f"ADR_{mesh_id}" if mesh_id else None

    # Get embeddings
    dvec = get_vec_by_key(d_key)
    avec = get_vec_by_key(a_key)
    gvec = get_vec_by_key(g_key) if g_key else np.zeros(dim)

    base_feats = np.concatenate([dvec, gvec, avec])

    # Cosine similarities
    cos_dg = safe_cosine(dvec, gvec)
    cos_ga = safe_cosine(gvec, avec)
    cos_da = safe_cosine(dvec, avec)

    # Optional literature features
    if literature_feats is not None and len(literature_feats) == len(nodes):
        def lit_vec_for(k): 
            if not k: return np.zeros(literature_feats.shape[1])
            if k in node_map:
                return literature_feats[node_map[k]]
            return np.zeros(literature_feats.shape[1])
        lid_d = lit_vec_for(d_key)
        lid_g = lit_vec_for(g_key)
        lid_a = lit_vec_for(a_key)
        lit_vec = np.mean([lid_d, lid_g, lid_a], axis=0)
    else:
        lit_vec = np.zeros(16)

    feature_vec = np.concatenate([base_feats, [cos_dg, cos_ga, cos_da], lit_vec])
    rows.append([d, g, a, mesh_id, row["label"]] + feature_vec.tolist())

# -------------------- SAVE OUTPUT --------------------
num_feats = len(rows[0]) - 5
cols = ["drug", "gene", "adr", "mesh_id", "label"] + [f"f{i}" for i in range(num_feats)]
outdf = pd.DataFrame(rows, columns=cols)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
outdf.to_csv(args.out, index=False)

print(f"💾 Saved fused feature table to {args.out}")
print(f"🧩 Shape: {outdf.shape}")
print("✅ Example rows:")
print(outdf.head(3))
