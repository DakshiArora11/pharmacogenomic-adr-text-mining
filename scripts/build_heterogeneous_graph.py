# build_heterogeneous_graph.py
import pandas as pd
import networkx as nx
from tqdm import tqdm

# Thresholds for adding similarity edges
DRUG_SIM_THRESHOLD = 0.3      # structure or CGI
ADR_SIM_THRESHOLD = 0.3       # MeSH or GDA

print("Loading drug and ADR lists...")
drugs = pd.read_csv("data/filtered_drugs_lincs.csv")   # must have 'pert_id' or 'DrugID'
adrs = pd.read_csv("data/filtered_adrs.csv")          # must have 'ADR_ID' column

drug_ids = drugs['pert_id'].astype(str).unique().tolist()
adr_ids = adrs['ADR_ID'].astype(str).unique().tolist()

# Load similarity matrices
print("Loading similarity matrices...")
drug_struct = pd.read_csv("data/drug_structure_similarity.csv", index_col=0)
drug_cgi = pd.read_csv("data/drug_cgi_similarity.csv", index_col=0)
adr_mesh = pd.read_csv("data/adr_mesh_similarity.csv", index_col=0)
adr_gda = pd.read_csv("data/adr_gda_similarity.csv", index_col=0)

# Load known drug–ADR associations (from filtered_adrs.csv)
# Make sure your filtered_adrs.csv has two columns: DrugID, ADR_ID
print("Loading known drug–ADR associations...")
known_pairs = adrs[['DrugID','ADR_ID']].astype(str).drop_duplicates()

# --- Build the graph ---
G = nx.Graph()

print("Adding drug nodes...")
for d in drug_ids:
    G.add_node(d, type='drug')

print("Adding ADR nodes...")
for a in adr_ids:
    G.add_node(a, type='adr')

print("Adding drug-drug similarity edges...")
for i, d1 in enumerate(drug_struct.index):
    for d2 in drug_struct.columns[i:]:
        if d1 in G and d2 in G:
            s1 = drug_struct.at[d1, d2] if d2 in drug_struct.columns else 0
            s2 = drug_cgi.at[d1, d2] if (d1 in drug_cgi.index and d2 in drug_cgi.columns) else 0
            score = max(s1, s2)
            if score > DRUG_SIM_THRESHOLD:
                G.add_edge(d1, d2, type='drug-drug', weight=score)

print("Adding ADR-ADR similarity edges...")
for i, a1 in enumerate(adr_mesh.index):
    for a2 in adr_mesh.columns[i:]:
        if a1 in G and a2 in G:
            s1 = adr_mesh.at[a1, a2] if a2 in adr_mesh.columns else 0
            s2 = adr_gda.at[a1, a2] if (a1 in adr_gda.index and a2 in adr_gda.columns) else 0
            score = max(s1, s2)
            if score > ADR_SIM_THRESHOLD:
                G.add_edge(a1, a2, type='adr-adr', weight=score)

print("Adding known drug-ADR edges...")
for _, row in tqdm(known_pairs.iterrows(), total=len(known_pairs)):
    d = row['MeSH_ID']
    a = row['SideEffectName']
    if d in G and a in G:
        G.add_edge(d, a, type='drug-adr', weight=1.0)

# Save edge list
edges_df = nx.to_pandas_edgelist(G)
edges_df.to_csv("data/heterogeneous_graph_edges.csv", index=False)
print("Saved ../data/heterogeneous_graph_edges.csv with", len(edges_df), "edges")
