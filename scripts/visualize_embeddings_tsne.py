import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

emb = np.load("data/dganet_literature_embeddings_fused_aligned.npy")
meta = pd.read_csv("data/dga_nodes.csv")

X = emb
labels = meta["Type"].values if "Type" in meta.columns else np.arange(len(X))

print(f"Embedding shape: {X.shape}")
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels, cmap="tab10", s=5)
plt.title("t-SNE of Fused Embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.show()
