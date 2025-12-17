import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

print("🔹 Loading embeddings...")

base_emb = np.load("data/dga_embeddings.npy")                  # baseline
lit_emb = np.load("data/dganet_literature_embeddings.npy")      # fused
nodes = pd.read_csv("data/dga_nodes.csv")

print(f"✅ Loaded baseline: {base_emb.shape}, literature: {lit_emb.shape}")

# --- Compute pairwise cosine similarities ---
sim_before = cosine_similarity(base_emb)
sim_after = cosine_similarity(lit_emb)

diff = sim_after - sim_before
mean_change = np.mean(np.abs(diff))
print(f"📊 Mean similarity change after literature fusion: {mean_change:.4f}")

# --- Identify top nodes with largest representation change ---
node_shift = np.linalg.norm(lit_emb - base_emb, axis=1)
top_changed = nodes.iloc[node_shift.argsort()[-10:][::-1]]
print("\n🔹 Top 10 nodes most influenced by literature:")
print(top_changed)

# --- Visualization ---
print("🔹 Generating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(lit_emb)

plt.figure(figsize=(10,8))
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=8, alpha=0.6)
plt.title("t-SNE of Literature-Augmented DGANet Embeddings")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.tight_layout()
plt.savefig("output/literature_embeddings_tsne.png", dpi=300)
plt.show()

print("✅ Visualization saved → output/literature_embeddings_tsne.png")
