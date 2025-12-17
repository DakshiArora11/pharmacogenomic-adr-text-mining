import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import difflib

tqdm.pandas()

print("🔹 Loading Drug–ADR, CTD, and SIDER drug name mapping...")

# --- Load SIDER drug–ADR pairs ---
drug_adr_edges = pd.read_csv("data/drug_adr_edges.csv")

# --- Load SIDER drug names ---
sider_names = pd.read_csv("data_raw/SIDER/drug_names.tsv", sep="\t", header=None, names=["STITCH_ID", "DrugName"])
sider_names["DrugName"] = sider_names["DrugName"].astype(str).str.lower()
sider_names["STITCH_ID"] = sider_names["STITCH_ID"].astype(str).str.replace("CID", "", regex=False)

print(f"✅ Loaded {len(sider_names)} valid SIDER drug names")

# --- Map SIDER drug IDs in edges to names ---
drug_adr_edges["DrugID"] = drug_adr_edges["Drug"].str.replace("DRUG_CID", "", regex=False)
drug_adr_edges = drug_adr_edges.merge(sider_names, left_on="DrugID", right_on="STITCH_ID", how="left")
print(f"✅ Mapped {drug_adr_edges['DrugName'].notna().sum()}/{len(drug_adr_edges)} drugs to real names")

# --- Load CTD drug–gene associations ---
ctd_cols = ["ChemicalName", "ChemicalID", "GeneSymbol", "GeneID", "InteractionActions", "PubMedIDs"]
ctd = pd.read_csv("data_raw/CTD/CTD_chem_gene_ixns.tsv.gz", sep="\t", comment="#", names=ctd_cols, low_memory=False)
ctd["ChemicalName"] = ctd["ChemicalName"].astype(str).str.lower()

print(f"✅ Loaded {len(ctd)} CTD drug–gene associations")

# --- Filter irrelevant CTD entries ---
ctd = ctd[ctd["ChemicalName"].str.len() > 3]
ctd = ctd[~ctd["ChemicalName"].str.contains("environment|pollutant|mixture|exposure|metal|gas", case=False)]

# --- Build name lists ---
sider_drug_names = sider_names["DrugName"].dropna().unique().tolist()

print("🔹 Matching CTD↔SIDER names (exact + substring + fuzzy)...")

def smart_match(ctd_name, sider_list):
    # exact match
    if ctd_name in sider_list:
        return ctd_name
    # substring match (e.g., "sodium acetate" → "acetate")
    for s in sider_list:
        if s in ctd_name or ctd_name in s:
            return s
    # fuzzy match
    matches = difflib.get_close_matches(ctd_name, sider_list, n=1, cutoff=0.7)
    return matches[0] if matches else None

ctd["MatchedDrug"] = ctd["ChemicalName"].progress_apply(lambda x: smart_match(x, sider_drug_names))

ctd_matched = ctd[ctd["MatchedDrug"].notna()]
print(f"✅ Matched {len(ctd_matched)} CTD drug–gene pairs after enhanced matching")

# --- Keep only drugs present in SIDER ADR list ---
ctd_filtered = ctd_matched[ctd_matched["MatchedDrug"].isin(drug_adr_edges["DrugName"].unique())]
print(f"✅ Filtered CTD: {len(ctd_filtered)} relevant drug–gene pairs after alignment")

# --- Build unified Drug–Gene–ADR graph ---
print("🔹 Building unified Drug–Gene–ADR graph...")
G = nx.Graph()

# Add Drug–ADR edges
for _, row in drug_adr_edges.iterrows():
    if pd.notna(row["DrugName"]):
        drug = f"DRUG_{row['DrugName']}"
        adr = row["ADR"]
        G.add_edge(drug, adr, relation="has_side_effect")

# Add Drug–Gene edges
for _, row in tqdm(ctd_filtered.iterrows(), total=len(ctd_filtered)):
    drug = f"DRUG_{row['MatchedDrug']}"
    gene = f"GENE_{row['GeneSymbol']}"
    G.add_edge(drug, gene, relation="targets_gene")

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Save outputs ---
edges_df = pd.DataFrame([(u, v, d["relation"]) for u, v, d in G.edges(data=True)], columns=["Source", "Target", "Relation"])
edges_df.to_csv("data/dga_edges.csv", index=False)
np.save("data/dga_adj.npy", nx.to_numpy_array(G))

print("💾 Saved DGA graph → data/dga_edges.csv and adjacency matrix")

# --- Summary ---
drug_nodes = [n for n in G.nodes if n.startswith("DRUG_")]
gene_nodes = [n for n in G.nodes if n.startswith("GENE_")]
adr_nodes = [n for n in G.nodes if n.startswith("ADR_")]

print(f"""
📊 DGA GRAPH SUMMARY
--------------------------
Total nodes: {G.number_of_nodes()}
Total edges: {G.number_of_edges()}
Drugs: {len(drug_nodes)}
Genes: {len(gene_nodes)}
ADRs: {len(adr_nodes)}
""")
