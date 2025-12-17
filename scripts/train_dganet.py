import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("🔹 Loading data...")

# --- Load features and node mappings ---
X = np.load("data/dga_features.npy")
nodes = pd.read_csv("data/dga_nodes.csv")["Node"].tolist()
node_to_idx = {n: i for i, n in enumerate(nodes)}

# --- Load edges ---
edges = pd.read_csv("data/dga_edges.csv")
edges_idx = np.array(
    [[node_to_idx[e.Source], node_to_idx[e.Target]] for _, e in edges.iterrows()]
).T

print(f"✅ Loaded {len(nodes)} nodes, {edges_idx.shape[1]} edges, feature dim = {X.shape[1]}")

# --- Prepare PyTorch Geometric data ---
x = torch.tensor(X, dtype=torch.float)
edge_index = torch.tensor(edges_idx, dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# -----------------------------------------------------------------------------
#  Define the GNN model (2-layer GraphSAGE)
# -----------------------------------------------------------------------------
from torch_geometric.nn import SAGEConv

class DGAnet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

# -----------------------------------------------------------------------------
#  Initialize and train the model
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DGAnet(in_channels=X.shape[1], hidden_channels=128, out_channels=64).to(device)
data = data.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss()

# Create synthetic link prediction task (positive + negative edges)
pos_edges = edges_idx.T
neg_edges = np.array([
    np.random.choice(len(nodes), size=2, replace=False) for _ in range(len(pos_edges))
]).T

train_edges, val_edges = train_test_split(pos_edges.T, test_size=0.2, random_state=42)
train_pos = torch.tensor(train_edges.T, dtype=torch.long).to(device)
val_pos = torch.tensor(val_edges.T, dtype=torch.long).to(device)

print("🔹 Training DGAnet model...")

for epoch in range(1, 51):
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index)

    # Link prediction: dot product of embeddings
    pos_score = (z[train_pos[0]] * z[train_pos[1]]).sum(dim=1)
    neg_score = (z[neg_edges[0]] * z[neg_edges[1]]).sum(dim=1)

    labels = torch.cat([torch.ones(len(pos_score)), torch.zeros(len(neg_score))]).to(device)
    preds = torch.cat([pos_score, neg_score])

    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# -----------------------------------------------------------------------------
#  Save learned embeddings
# -----------------------------------------------------------------------------
model.eval()
embeddings = model(data.x, data.edge_index).detach().cpu().numpy()
np.save("data/dga_embeddings.npy", embeddings)

print("💾 Saved embeddings → data/dga_embeddings.npy")
print("✅ Training complete!")
