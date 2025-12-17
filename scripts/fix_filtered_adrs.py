import pandas as pd
import gzip
from rapidfuzz import process, fuzz

# --- Load SIDER ADRs ---
sider_path = "../data_raw/SIDER/meddra_all_se.tsv.gz"
sider = pd.read_csv(sider_path, sep="\t", header=None,
                    names=["STITCH_ID", "UMLS_ID", "MedDRA_type", "SideEffectName", "Frequency"])

# Keep only unique ADRs (UMLS ID + name)
sider = sider[["UMLS_ID", "SideEffectName"]].drop_duplicates()

print("Loaded SIDER ADRs:", sider.shape)

# --- Load MeSH descriptors ---
mesh_terms = {}
with open("../data_raw/MeSH/desc2025.txt", "r", encoding="utf-8", errors="ignore") as f:
    current_id, current_name = None, None
    for line in f:
        if line.startswith("UI = "):  # Descriptor Unique ID
            current_id = "MESH:" + line.strip().split(" = ")[1]
        elif line.startswith("MH = "):  # Descriptor Name
            current_name = line.strip().split(" = ")[1]
            if current_id and current_name:
                mesh_terms[current_id] = current_name
                current_id, current_name = None, None

print("Loaded MeSH terms:", len(mesh_terms))

# --- Build list for fuzzy matching ---
mesh_names = list(mesh_terms.values())
mesh_ids = list(mesh_terms.keys())

def map_to_mesh(name):
    """Find best matching MeSH term for an ADR name"""
    if pd.isna(name):
        return None, None
    match, score, idx = process.extractOne(
        name, mesh_names, scorer=fuzz.token_sort_ratio
    )
    if score > 80:  # only keep strong matches
        return mesh_ids[idx], match
    else:
        return None, None

# --- Match ADR names ---
mapped = []
for _, row in sider.iterrows():
    umls = row["UMLS_ID"]
    name = row["SideEffectName"]
    mesh_id, mesh_name = map_to_mesh(name)
    if mesh_id:
        mapped.append((mesh_id, mesh_name))

df_mapped = pd.DataFrame(mapped, columns=["MeSH_ID", "SideEffectName"]).drop_duplicates()

print("Mapped ADRs:", df_mapped.shape)

# --- Save fixed ADRs ---
df_mapped.to_csv("../data/filtered_adrs.csv", index=False)
print("✅ Saved corrected ADRs to ../data/filtered_adrs.csv")
