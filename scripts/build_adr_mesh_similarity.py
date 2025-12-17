import pandas as pd
import numpy as np
import re
from tqdm import tqdm

# --- Load filtered ADRs ---
adrs = pd.read_csv("data/filtered_adrs.csv")
adrs = adrs.dropna(subset=["MeSH_ID"])
mesh_ids = adrs["MeSH_ID"].unique().tolist()

print(f"✅ Loaded {len(mesh_ids)} unique MeSH ADR IDs")

# --- Parse MeSH descriptors from text dump ---
mesh_path = "data_raw/MESH/desc2025_fixed.txt"
mesh_records = {}

with open(mesh_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Find all MeSH IDs (like D000001) and their tree numbers (like D27.505.954.122)
entries = re.findall(r"(D\d{6})(.*?)D\d{6}|(D\d{6}).*$", text, flags=re.S)

for match in entries:
    # Each match is either (id, content, _) or (_, _, id)
    mesh_id = match[0] or match[2]
    if not mesh_id:
        continue
    block = match[1]
    trees = re.findall(r"(?:[A-Z]\d{2}(?:\.\d{3,})+)", block)
    if trees:
        mesh_records[mesh_id] = {"TreeNumbers": trees}

print(f"✅ Parsed {len(mesh_records)} MeSH descriptors with hierarchy info")

# --- Compute MeSH semantic similarity ---
def mesh_similarity(mesh_a, mesh_b):
    if mesh_a not in mesh_records or mesh_b not in mesh_records:
        return 0.0
    trees_a = mesh_records[mesh_a]["TreeNumbers"]
    trees_b = mesh_records[mesh_b]["TreeNumbers"]
    max_sim = 0.0
    for ta in trees_a:
        for tb in trees_b:
            a_split = ta.split(".")
            b_split = tb.split(".")
            common = 0
            for a, b in zip(a_split, b_split):
                if a == b:
                    common += 1
                else:
                    break
            sim = common / max(len(a_split), len(b_split))
            max_sim = max(max_sim, sim)
    return max_sim

# --- Build similarity matrix ---
sim = np.zeros((len(mesh_ids), len(mesh_ids)))

for i, a in enumerate(tqdm(mesh_ids, desc="Computing MeSH similarities")):
    for j, b in enumerate(mesh_ids):
        if i <= j:
            val = mesh_similarity(a, b)
            sim[i, j] = sim[j, i] = val

sim_df = pd.DataFrame(sim, index=mesh_ids, columns=mesh_ids)
sim_df.to_csv("data/adr_mesh_similarity.csv")

non_zero = np.count_nonzero(sim)
print(f"✅ Saved data/adr_mesh_similarity.csv with shape {sim_df.shape}")
print(f"Non-zero entries: {non_zero}")
