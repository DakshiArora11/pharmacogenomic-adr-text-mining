import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

# Config
nodes_path = "data/dga_nodes_expanded.csv"
edges_path = "data/kg_edges.csv"
out_path = "data/dganet_literature_embeddings_contrastive.npy"
embedding_dim = 64
epochs = 100
lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using device: {device}")
print("📥 Loading graph data...")

nodes = pd.read_csv(nodes_path)
edges = pd.read_csv(edges_path)
node_to_idx = {n: i for i, n in enumerate(nodes["Node"])}
num_nodes = len(nodes)
print(f"🧩 Total nodes: {num_nodes} | Edges: {len(edges)}")

# Build graph
src = edges["src"].map(node_to_idx)
dst = edges["dst"].map(node_to_idx)
edge_index = torch.tensor([src.values, dst.values], dtype=torch.long)

# Node features = identity
x = torch.eye(num_nodes, dtype=torch.float32)

data = Data(x=x, edge_index=edge_index).to(device)

# --- GNN model ---
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, data):
        x = self.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        return x

model = GraphSAGE(num_nodes, 128, embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# --- Contrastive loss ---
cosine = nn.CosineSimilarity(dim=1)
bce = nn.BCEWithLogitsLoss()

def sample_negative_edges(num_neg, num_nodes):
    src = torch.randint(0, num_nodes, (num_neg,))
    dst = torch.randint(0, num_nodes, (num_neg,))
    return torch.stack([src, dst], dim=0)

print(f"🧠 Training contrastive GNN ({epochs} epochs)...")

for epoch in tqdm(range(1, epochs + 1)):
    model.train()
    optimizer.zero_grad()

    z = model(data)
    src_pos, dst_pos = edge_index[0], edge_index[1]
    pos_sim = cosine(z[src_pos], z[dst_pos])
    pos_label = torch.ones_like(pos_sim)

    neg_edges = sample_negative_edges(len(src_pos), num_nodes)
    neg_sim = cosine(z[neg_edges[0]], z[neg_edges[1]])
    neg_label = torch.zeros_like(neg_sim)

    loss = bce(torch.cat([pos_sim, neg_sim]), torch.cat([pos_label, neg_label]))

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

model.eval()
emb = model(data).detach().cpu().numpy()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.save(out_path, emb)
print(f"💾 Saved → {out_path}")
print("✅ Embedding shape:", emb.shape)
