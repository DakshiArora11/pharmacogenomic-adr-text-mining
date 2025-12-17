import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

print("🔹 Loading DGA edges and building graph...")

# --- Load the DGA edge list ---
edges = pd.read_csv("data/dga_edges.csv")
G = nx.from_pandas_edgelist(edges, "Source", "Target", edge_attr=True)

print(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# --- Extract node types ---
drugs = [n for n in G.nodes if n.startswith("DRUG_")]
genes = [n for n in G.nodes if n.startswith("GENE_")]
adrs = [n for n in G.nodes if n.startswith("ADR_")]

print(f"Drugs: {len(drugs)}, Genes: {len(genes)}, ADRs: {len(adrs)}")

# -----------------------------------------------------------------------------
# 1️⃣ ADR FEATURES (from MeSH Names)
# -----------------------------------------------------------------------------
print("🔹 Building ADR features using MeSH names...")

filtered_adrs = pd.read_csv("data/filtered_adrs.csv")[["MeSH_ID", "MeSH_Name"]].dropna()
adr_texts = {f"ADR_{row.MeSH_ID}": row.MeSH_Name for _, row in filtered_adrs.iterrows()}

adr_names = [adr_texts.get(a, "") for a in adrs]
vectorizer = TfidfVectorizer(max_features=300)
adr_features = vectorizer.fit_transform(adr_names).toarray()
adr_features = normalize(adr_features)

print(f"✅ ADR feature matrix: {adr_features.shape}")

# -----------------------------------------------------------------------------
# 2️⃣ GENE FEATURES (from gene symbols - one-hot)
# -----------------------------------------------------------------------------
print("🔹 Building GENE features (one-hot)...")

unique_genes = sorted(list(set([g.replace("GENE_", "") for g in genes])))
gene_to_idx = {g: i for i, g in enumerate(unique_genes)}

gene_features = np.eye(len(unique_genes))
gene_feat_dict = {f"GENE_{g}": gene_features[i] for i, g in enumerate(unique_genes)}

print(f"✅ Gene feature matrix: {gene_features.shape}")

# -----------------------------------------------------------------------------
# 3️⃣ DRUG FEATURES (from name embeddings via TF-IDF)
# -----------------------------------------------------------------------------
print("🔹 Building DRUG features from names...")

sider_names = pd.read_csv("data_raw/SIDER/drug_names.tsv", sep="\t", header=None, names=["STITCH_ID", "DrugName"])
sider_names["DrugName"] = sider_names["DrugName"].astype(str).str.lower()
drug_name_map = {f"DRUG_{name}": name for name in sider_names["DrugName"].unique()}

drug_texts = [drug_name_map.get(d, "") for d in drugs]
drug_vectorizer = TfidfVectorizer(max_features=300)
drug_features = drug_vectorizer.fit_transform(drug_texts).toarray()
drug_features = normalize(drug_features)

print(f"✅ Drug feature matrix: {drug_features.shape}")

# -----------------------------------------------------------------------------
# Combine all node features
# -----------------------------------------------------------------------------
print("🔹 Combining node features...")

node_features = {}
for i, d in enumerate(drugs):
    node_features[d] = drug_features[i]
for i, a in enumerate(adrs):
    node_features[a] = adr_features[i]
for g in genes:
    node_features[g] = gene_feat_dict.get(g, np.zeros(len(unique_genes)))

# Stack into consistent matrix
all_nodes = list(G.nodes)
feature_dim = len(next(iter(node_features.values())))
X = np.zeros((len(all_nodes), feature_dim))

for i, node in enumerate(all_nodes):
    if node in node_features:
        vec = node_features[node]
        if len(vec) != feature_dim:
            vec = np.pad(vec, (0, feature_dim - len(vec)))
        X[i] = vec

# --- Save features ---
np.save("data/dga_features.npy", X)
pd.DataFrame({"Node": all_nodes}).to_csv("data/dga_nodes.csv", index=False)

print(f"💾 Saved feature matrix → data/dga_features.npy ({X.shape})")
print(f"💾 Saved node list → data/dga_nodes.csv")

print("\n✅ DGA feature generation complete!")
