import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# ----------------------
# Config
# ----------------------
NODES_CSV = "data/dga_nodes_expanded.csv"
EDGES_CSV = "data/kg_edges.csv"          # same as before (drug-gene, gene-adr, etc.)
OUT_NPY   = "data/dganet_literature_embeddings_contrastive_v2.npy"

HIDDEN = 128
EMB_DIM = 64
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 4096
TEMP = 0.07          # InfoNCE temperature
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"🚀 Device: {device}")
print("📥 Loading graph...")

nodes = pd.read_csv(NODES_CSV)
edges = pd.read_csv(EDGES_CSV)

node_to_idx = {n: i for i, n in enumerate(nodes["Node"])}
num_nodes = len(nodes)
print(f"🧩 Nodes: {num_nodes:,} | Edges: {len(edges):,}")

# Build edge index (undirected helps neighborhood)
src = edges["src"].map(node_to_idx).astype(int)
dst = edges["dst"].map(node_to_idx).astype(int)
edge_index = torch.tensor(np.vstack([src.values, dst.values]), dtype=torch.long)

# Node features: identity for self-supervised
x = torch.eye(num_nodes, dtype=torch.float32)
data = Data(x=x, edge_index=edge_index).to(device)

# ----------------------
# Hard negative pools
# ----------------------
# Heuristics: group nodes by coarse type (DRUG_, GENE_, ADR_) to pick "harder" negatives from same type.
node_labels = nodes["Node"].tolist()
type_groups = {"DRUG": [], "GENE": [], "ADR": [], "OTHER": []}
for i, name in enumerate(node_labels):
    if name.startswith("DRUG_"):
        type_groups["DRUG"].append(i)
    elif name.startswith("GENE_"):
        type_groups["GENE"].append(i)
    elif name.startswith("ADR_"):
        type_groups["ADR"].append(i)
    else:
        type_groups["OTHER"].append(i)

type_groups = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in type_groups.items()}

def node_type(idx: torch.Tensor):
    # very light heuristic by name prefix
    names = [node_labels[i] for i in idx.tolist()]
    t = []
    for n in names:
        if n.startswith("DRUG_"): t.append("DRUG")
        elif n.startswith("GENE_"): t.append("GENE")
        elif n.startswith("ADR_"):  t.append("ADR")
        else:                       t.append("OTHER")
    return t

def sample_hard_negatives(src_batch, dst_batch):
    """
    For each (u,v) in batch, pick a negative (u, v') where v' shares node-type with v (harder),
    and also pick (u', v) where u' shares node-type with u. Then concatenate.
    """
    with torch.no_grad():
        s_types = node_type(src_batch)
        d_types = node_type(dst_batch)

        # pick v' from same type as v (if pool empty, fallback to uniform)
        v_prime = []
        for t in d_types:
            pool = type_groups.get(t, None)
            if pool is not None and len(pool) > 0:
                v_prime.append(pool[torch.randint(0, len(pool), (1,))])
            else:
                v_prime.append(torch.randint(0, num_nodes, (1,), device=device))
        v_prime = torch.stack(v_prime).squeeze(-1)

        # pick u' from same type as u
        u_prime = []
        for t in s_types:
            pool = type_groups.get(t, None)
            if pool is not None and len(pool) > 0:
                u_prime.append(pool[torch.randint(0, len(pool), (1,))])
            else:
                u_prime.append(torch.randint(0, num_nodes, (1,), device=device))
        u_prime = torch.stack(u_prime).squeeze(-1)

    # negatives: (u, v') and (u', v)
    neg_uvp = torch.stack([src_batch, v_prime], dim=0)
    neg_upv = torch.stack([u_prime, dst_batch], dim=0)
    return neg_uvp, neg_upv

# ----------------------
# Model
# ----------------------
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)
        self.act = nn.ReLU()

    def forward(self, data):
        x = self.act(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        x = nn.functional.normalize(x, p=2, dim=-1)  # L2 normalize for cosine/InfoNCE
        return x

model = GraphSAGE(num_nodes, HIDDEN, EMB_DIM).to(device)
opt = optim.Adam(model.parameters(), lr=LR)

# ----------------------
# InfoNCE loss
# ----------------------
def info_nce(anchor, positive, negatives, temperature=0.07):
    """
    anchor:  (B, D)
    positive:(B, D)
    negatives:(K*B, D) — multiple negatives per anchor (we'll use K=2)
    """
    # Cosine similarity
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # (B,)
    # For each anchor, we tile it to match negatives
    B = anchor.size(0)
    neg = negatives.view(2, B, -1)   # 2 negatives of shape (B, D) stacked => reshape to (2,B,D)
    neg = neg.reshape(-1, anchor.size(1))  # (2B, D)
    anchor_tiled = anchor.repeat(2, 1)     # (2B, D)
    neg_sim = torch.sum(anchor_tiled * neg, dim=-1) / temperature  # (2B,)

    # Log-softmax over (pos + negatives) per anchor
    # Build logits: for each i in batch, logits = [pos_i, neg_i1, neg_i2]
    logits = []
    for i in range(B):
        # pos logit
        lp = pos_sim[i].unsqueeze(0)
        # two neg logits belonging to this i: indices i and i+B in neg_sim
        ln1 = neg_sim[i]
        ln2 = neg_sim[i + B]
        logits.append(torch.stack([lp.squeeze(0), ln1, ln2], dim=0))
    logits = torch.stack(logits, dim=0)  # (B, 3)

    labels = torch.zeros(B, dtype=torch.long, device=logits.device)  # index 0 is positive
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

# ----------------------
# Training (mini-batch over edges)
# ----------------------
num_edges = edge_index.size(1)
perm = torch.randperm(num_edges)
print(f"🧠 Training contrastive GNN v2 ({EPOCHS} epochs, InfoNCE, hard negatives)...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    opt.zero_grad()
    epoch_loss = 0.0

    # Recompute embeddings each mini-batch to avoid reusing autograd graph
    for i in range(0, num_edges, BATCH_SIZE):
        j = min(i + BATCH_SIZE, num_edges)
        batch_idx = perm[i:j]

        # Compute embeddings fresh inside batch to avoid graph reuse
        z = model(data)
        src_b = edge_index[0, batch_idx]
        dst_b = edge_index[1, batch_idx]

        # Positives
        a = z[src_b]
        p = z[dst_b]

        # Hard negatives
        neg_uvp, neg_upv = sample_hard_negatives(src_b, dst_b)
        n1 = z[neg_uvp[1]]  # v'
        n2 = z[neg_upv[0]]  # u'
        negatives = torch.cat([n1, n2], dim=0)

        loss = info_nce(a, p, negatives, temperature=TEMP)
        loss.backward()
        opt.step()
        opt.zero_grad()  # clear grads between mini-batches
        epoch_loss += loss.item()

    avg = epoch_loss / math.ceil(num_edges / BATCH_SIZE)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | InfoNCE loss: {avg:.4f}")
