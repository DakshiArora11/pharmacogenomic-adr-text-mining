import pandas as pd
import gzip
from tqdm import tqdm
import numpy as np

# --- Load filtered ADRs ---
adrs = pd.read_csv("data/filtered_adrs.csv")
adrs = adrs.dropna(subset=["MeSH_ID"])
adr_mesh_ids = adrs["MeSH_ID"].unique().tolist()
adr_mesh_set = set(adr_mesh_ids)
print(f"✅ Loaded {len(adr_mesh_ids)} ADRs")

# --- Build mapping MeSH_ID → genes (streaming CTD file) ---
ctd_path = "data_raw/CTD/CTD_genes_diseases.tsv.gz"
disease_to_genes = {}

with gzip.open(ctd_path, "rt", encoding="utf-8", errors="ignore") as f:
    for line in tqdm(f, desc="Streaming CTD data"):
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        gene_symbol = parts[0].strip()
        disease_id = parts[3].strip()
        if disease_id.startswith("MESH:"):
            mesh_id = disease_id.replace("MESH:", "")
            if mesh_id in adr_mesh_set:  # keep only relevant ADRs
                if mesh_id not in disease_to_genes:
                    disease_to_genes[mesh_id] = set()
                disease_to_genes[mesh_id].add(gene_symbol)

print(f"✅ Built mapping for {len(disease_to_genes)} ADRs linked to genes")

# --- Compute Jaccard similarities ---
print("🧮 Computing Jaccard similarities...")
sim = np.zeros((len(adr_mesh_ids), len(adr_mesh_ids)), dtype=np.float32)

for i, a in enumerate(tqdm(adr_mesh_ids)):
    genes_a = disease_to_genes.get(a, set())
    for j in range(i, len(adr_mesh_ids)):
        b = adr_mesh_ids[j]
        genes_b = disease_to_genes.get(b, set())
        if not genes_a or not genes_b:
            val = 0.0
        else:
            intersection = len(genes_a & genes_b)
            union = len(genes_a | genes_b)
            val = intersection / union if union > 0 else 0.0
        sim[i, j] = sim[j, i] = val

# --- Save results ---
sim_df = pd.DataFrame(sim, index=adr_mesh_ids, columns=adr_mesh_ids)
out_path = "data/adr_gda_similarity.csv"
sim_df.to_csv(out_path)
print(f"✅ Saved {out_path} with shape {sim_df.shape}")

# --- Stats ---
non_zero = np.count_nonzero(sim)
print(f"Non-zero entries: {non_zero}")
linked_adrs = sum(1 for a in adr_mesh_ids if a in disease_to_genes)
print(f"ADRs with CTD gene links: {linked_adrs}/{len(adr_mesh_ids)}")
