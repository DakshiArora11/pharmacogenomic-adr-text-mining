import pandas as pd
import numpy as np
import networkx as nx

# ============================================================
# 1️⃣ Load Data
# ============================================================
print("🔹 Loading SIDER and filtered ADRs...")

sider_file = "data_raw/SIDER/meddra_all_se.tsv.gz"
adrs_file = "data/filtered_adrs.csv"

sider = pd.read_csv(sider_file, sep="\t", header=None, low_memory=False)
adrs = pd.read_csv(adrs_file)

sider.columns = [
    "STITCH_ID_flat",
    "STITCH_ID_stereo",
    "UMLS_ID",
    "MedDRA_type",
    "SideEffectName",
    "Frequency"
]

print(f"✅ Loaded {len(sider)} SIDER entries and {len(adrs)} filtered ADRs")

# ============================================================
# 2️⃣ Merge SIDER → filtered ADRs to get MeSH IDs
# ============================================================
merged = sider.merge(adrs[["UMLS_ID", "MeSH_ID", "MeSH_Name"]],
                     left_on="UMLS_ID", right_on="UMLS_ID", how="inner")

print(f"✅ Mapped {len(merged)} drug–ADR pairs to MeSH")

# ============================================================
# 3️⃣ Build Bipartite Graph
# ============================================================
print("🔹 Building Drug–ADR bipartite graph...")

G = nx.Graph()

# Add drug nodes (prefix 'DRUG_') and ADR nodes (prefix 'ADR_')
for _, row in merged.iterrows():
    drug = f"DRUG_{row['STITCH_ID_flat']}"
    adr = f"ADR_{row['MeSH_ID']}"
    G.add_node(drug, bipartite=0)
    G.add_node(adr, bipartite=1)
    G.add_edge(drug, adr)

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# ============================================================
# 4️⃣ Extract Bipartite Matrices
# ============================================================
drug_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
adr_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

adj_matrix = nx.to_pandas_adjacency(G, nodelist=drug_nodes + adr_nodes)
np.save("data/drug_adr_adj.npy", adj_matrix.values)

# Save edge list
edges_df = pd.DataFrame([(u, v) for u, v in G.edges()], columns=["Drug", "ADR"])
edges_df.to_csv("data/drug_adr_edges.csv", index=False)

print("💾 Saved adjacency matrix and edge list")

# ============================================================
# 5️⃣ Summary Stats
# ============================================================
drug_count = len(drug_nodes)
adr_count = len(adr_nodes)
print(f"""
📊 BIPARTITE GRAPH SUMMARY
--------------------------
Drugs: {drug_count}
ADRs: {adr_count}
Edges (associations): {G.number_of_edges()}
Average degree (drugs): {np.mean([G.degree(d) for d in drug_nodes]):.2f}
Average degree (ADRs): {np.mean([G.degree(a) for a in adr_nodes]):.2f}
""")
