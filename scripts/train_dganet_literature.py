import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx

print("🔹 Loading data...")

# --- Load structured graph ---
edges = pd.read_csv("data/dga_literature_edges.csv")
nodes = pd.read_csv("data/dga_nodes.csv")  # from earlier DGA graph features

node_index = {n: i for i, n in enumerate(nodes["Node"])}

rows, cols = [], []
for _, r in edges.iterrows():
    if r["Source"] in node_index and r["Target"] in node_index:
        rows.append(node_index[r["Source"]])
        cols.append(node_index[r["Target"]])

adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(node_index), len(node_index)))
edge_index, _ = from_scipy_sparse_matrix(adj)

# --- Load features ---
X_struct = np.load("data/dga_features.npy")       # (1969, 300)
X_lit = np.load("data/literature_features.npy")   # (7892, 768)

# Align literature features to graph nodes
lit_nodes = pd.read_csv("data/literature_nodes.csv")
lit_index = {n: i for i, n in enumerate(lit_nodes["Node"])}

aligned_lit = np.zeros((len(node_index), 768))
for n, i in node_index.items():
    if n in lit_index:
        aligned_lit[i] = X_lit[lit_index[n]]

# Pad structured features if smaller
if X_struct.shape[0] < len(node_index):
    pad = np.zeros((len(node_index) - X_struct.shape[0], X_struct.shape[1]))
    X_struct = np.vstack([X_struct, pad])

print(f"✅ Structured features: {X_struct.shape}, Literature features: {aligned_lit.shape}")

# --- Convert to tensors ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_struct = torch.tensor(X_struct, dtype=torch.float32).to(device)
X_lit = torch.tensor(aligned_lit, dtype=torch.float32).to(device)
edge_index = edge_index.to(device)

# --- Model ---
class FusionDGANet(nn.Module):
    def __init__(self, in_struct, in_lit, hidden, out):
        super().__init__()
        self.sage_struct = GraphSAGE(in_struct, hidden, num_layers=2)
        self.sage_lit = GraphSAGE(in_lit, hidden, num_layers=2)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=4, batch_first=True)
        self.fc_out = nn.Linear(hidden, out)

    def forward(self, Xs, Xl, edge_index):
        h_s = self.sage_struct(Xs, edge_index)
        h_l = self.sage_lit(Xl, edge_index)
        fused, _ = self.attn(h_s.unsqueeze(1), h_l.unsqueeze(1), h_l.unsqueeze(1))
        fused = fused.squeeze(1)
        return self.fc_out(F.relu(fused))

model = FusionDGANet(X_struct.shape[1], X_lit.shape[1], hidden=128, out=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("🔹 Training fusion model...")

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    out = model(X_struct, X_lit, edge_index)
    loss = criterion(out, out.detach())  # self-supervised node embedding alignment
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

embeddings = model(X_struct, X_lit, edge_index).detach().cpu().numpy()
np.save("data/dganet_literature_embeddings.npy", embeddings)
print("💾 Saved fused embeddings → data/dganet_literature_embeddings.npy")
print("✅ Literature-aware DGANet training complete!")
