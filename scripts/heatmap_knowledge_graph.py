import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load edges
edges = pd.read_csv("data/kg_edges.csv")

# Ensure columns exist
if "src" not in edges.columns or "dst" not in edges.columns:
    raise ValueError("CSV must contain 'src' and 'dst' columns")

# Convert IDs to categorical → index → numeric IDs
edges["src_id"] = edges["src"].astype("category").cat.codes
edges["dst_id"] = edges["dst"].astype("category").cat.codes

# Build adjacency matrix for first N nodes
N = 200   # change this based on what you want to visualize
unique_nodes = max(edges["src_id"].max(), edges["dst_id"].max()) + 1
N = min(N, unique_nodes)

# Initialize matrix
adj = np.zeros((N, N))

# Populate adjacency for src→dst
for _, row in edges.iterrows():
    s = row["src_id"]
    d = row["dst_id"]
    if s < N and d < N:
        adj[s, d] = 1

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(adj, cmap="viridis", square=True, cbar_kws={"label": "Edge Presence"})
plt.title("Knowledge Graph Adjacency Heatmap (First 200 Nodes)")
plt.xlabel("Node Index")
plt.ylabel("Node Index")
plt.tight_layout()
plt.show()
