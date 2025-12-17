import pandas as pd
import numpy as np
import networkx as nx

# ============================================================
# 1️⃣ Load fused similarity matrix
# ============================================================
print("🔹 Loading fused similarity matrix...")
sim_df = pd.read_csv("data/adr_fused_similarity.csv", index_col=0)
print(f"✅ Loaded fused matrix with shape {sim_df.shape}")

# ============================================================
# 2️⃣ Define similarity threshold
# ============================================================
# Keep only strong connections (you can tune this)
THRESHOLD = 0.2

# ============================================================
# 3️⃣ Create graph using NetworkX
# ============================================================
print(f"🔹 Building graph (edges where similarity ≥ {THRESHOLD})...")

G = nx.Graph()

# Add nodes
for node in sim_df.index:
    G.add_node(node)

# Add edges (symmetric, thresholded)
values = sim_df.values
n = len(sim_df)
for i in range(n):
    for j in range(i + 1, n):
        sim_val = values[i, j]
        if sim_val >= THRESHOLD:
            G.add_edge(sim_df.index[i], sim_df.index[j], weight=float(sim_val))

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# ============================================================
# 4️⃣ Save adjacency matrix
# ============================================================
adj_matrix = nx.to_numpy_array(G, nodelist=sim_df.index)
np.save("data/adr_graph_adj.npy", adj_matrix)
print(f"💾 Saved adjacency matrix → data/adr_graph_adj.npy")

# ============================================================
# 5️⃣ Save edge list (for inspection)
# ============================================================
edges_df = pd.DataFrame([(u, v, d["weight"]) for u, v, d in G.edges(data=True)],
                        columns=["Source", "Target", "Weight"])
edges_df.to_csv("data/adr_graph_edges.csv", index=False)
print(f"💾 Saved edge list → data/adr_graph_edges.csv")

# ============================================================
# 6️⃣ Basic graph stats
# ============================================================
avg_deg = np.mean([d for n, d in G.degree()])
density = nx.density(G)
components = nx.number_connected_components(G)

print(f"""
📊 GRAPH SUMMARY
----------------------
Nodes: {G.number_of_nodes()}
Edges: {G.number_of_edges()}
Average Degree: {avg_deg:.2f}
Density: {density:.4f}
Connected Components: {components}
""")
