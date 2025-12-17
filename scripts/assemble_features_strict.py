import pandas as pd, numpy as np, os
from tqdm import tqdm

print("📥 Loading data...")

triples = pd.read_csv("data/triples_normalized.csv")
nodes = pd.read_csv("data/dga_nodes_expanded.csv")
embeddings = np.load("data/dganet_literature_embeddings_fused_aligned.npy")

node_map = {n: i for i, n in enumerate(nodes["Node"])}

def get_vec(node):
    idx = node_map.get(node)
    if idx is None:
        return np.random.normal(0, 0.01, embeddings.shape[1])
    return embeddings[idx]

def clean(s): return str(s).strip().replace(" ", "_")

rows = []
for _, r in tqdm(triples.iterrows(), total=len(triples)):
    d, g, a = clean(r["Drug"]), clean(r["Gene"]), clean(r["ADR"])
    mesh = r.get("MeSH_ID", "")

    dvec, gvec, avec = get_vec(f"DRUG_{d.lower()}"), get_vec(f"GENE_{g.upper()}"), get_vec(f"ADR_{a}")
    feat = np.concatenate([dvec, gvec, avec])
    rows.append([d, g, a, mesh] + feat.tolist())

header = ["drug", "gene", "adr", "mesh_id"] + [f"f{i}" for i in range(len(rows[0]) - 4)]
df = pd.DataFrame(rows, columns=header)

os.makedirs("output", exist_ok=True)
df.to_csv("output/features_fused_strict.csv", index=False)
print("✅ Saved → output/features_fused_strict.csv")
print("📊 Shape:", df.shape)
