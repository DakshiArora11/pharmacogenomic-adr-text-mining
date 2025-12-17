import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re

print("🔹 Loading literature triples and graph nodes...")

triples = pd.read_csv("data/triples_normalized.csv")
edges = pd.read_csv("data/dga_literature_edges.csv")

# Extract all unique node IDs from the graph
all_nodes = set(edges["Source"]).union(set(edges["Target"]))
print(f"✅ Loaded {len(all_nodes)} unique graph nodes")

# --- Prepare sentences to represent each node ---
def clean_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", str(x)).strip().lower()

sentences = {}
for node in all_nodes:
    if node.startswith("DRUG_"):
        name = node.replace("DRUG_", "")
        text = f"The drug {name} is associated with multiple genes and adverse reactions."
    elif node.startswith("GENE_"):
        name = node.replace("GENE_", "")
        text = f"The gene {name} has pharmacogenomic interactions with various drugs and side effects."
    elif node.startswith("ADR_"):
        name = node.replace("ADR_", "")
        text = f"The adverse reaction {name} is reported in literature and linked to certain genes and drugs."
    else:
        text = node
    sentences[node] = clean_text(text)

print(f"✅ Generated sentences for {len(sentences)} nodes")

# --- Load literature model ---
print("🔹 Generating embeddings using BioBERT (via SentenceTransformer)...")
model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

embeddings = []
node_list = []

for node in tqdm(all_nodes):
    text = sentences[node]
    emb = model.encode(text)
    embeddings.append(emb)
    node_list.append(node)

embeddings = np.array(embeddings)
np.save("data/literature_features.npy", embeddings)
pd.DataFrame({"Node": node_list}).to_csv("data/literature_nodes.csv", index=False)

print(f"💾 Saved literature embeddings → data/literature_features.npy ({embeddings.shape})")
print("✅ Literature feature generation complete!")
