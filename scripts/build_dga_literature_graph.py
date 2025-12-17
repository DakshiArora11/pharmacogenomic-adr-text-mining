import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

print("🔹 Loading structured DGA graph...")
dga_edges = pd.read_csv("data/dga_edges.csv")
print(f"✅ Loaded structured DGA edges: {len(dga_edges)}")

print("🔹 Loading literature triples...")
lit_triples = pd.read_csv("data/triples_normalized.csv")
print(f"✅ Loaded normalized literature triples: {len(lit_triples)}")

# --- Build base graph ---
G = nx.Graph()

# Add structured DGA edges
for _, row in dga_edges.iterrows():
    G.add_edge(row["Source"], row["Target"], relation=row["Relation"])

# Add literature triples
for _, row in tqdm(lit_triples.iterrows(), total=len(lit_triples)):
    drug = f"DRUG_CID{int(row['DrugID'])}"
    gene = f"GENE_{row['Gene']}"
    adr = f"ADR_{row['MeSH_ID']}"

    G.add_edge(drug, gene, relation="lit_targets_gene")
    G.add_edge(gene, adr, relation="lit_associated_with_adr")
    G.add_edge(drug, adr, relation="lit_causes_adr")

print(f"✅ Unified graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Save outputs ---
edges_df = pd.DataFrame(
    [(u, v, d["relation"]) for u, v, d in G.edges(data=True)],
    columns=["Source", "Target", "Relation"]
)
edges_df.to_csv("data/dga_literature_edges.csv", index=False)
np.save("data/dga_literature_adj.npy", nx.to_numpy_array(G))

print("💾 Saved enriched graph → data/dga_literature_edges.csv and adjacency matrix")

# --- Summary ---
drug_nodes = [n for n in G.nodes if n.startswith("DRUG_")]
gene_nodes = [n for n in G.nodes if n.startswith("GENE_")]
adr_nodes = [n for n in G.nodes if n.startswith("ADR_")]

print(f"""
📊 LITERATURE-AUGMENTED DGA GRAPH SUMMARY
-----------------------------------------
Total nodes: {G.number_of_nodes()}
Total edges: {G.number_of_edges()}
Drugs: {len(drug_nodes)}
Genes: {len(gene_nodes)}
ADRs: {len(adr_nodes)}
""")
