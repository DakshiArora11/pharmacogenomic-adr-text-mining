import numpy as np, pandas as pd, pickle, torch, re
from sklearn.preprocessing import StandardScaler

EMB = "data/dganet_literature_embeddings_fused_aligned.npy"
NODES = "data/dga_nodes_expanded.csv"
KG = "data/kg_edges.csv"
FEAT_COLS = "models/triple_scorer_scaler.npy"
MODEL = "models/triple_scorer.pt"
CAL = "models/calibrator_isotonic.pkl"

import torch.nn as nn
class Scorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

def norm_drug(s): return str(s).strip().lower().replace(" ", "_")
def norm_gene(s): return str(s).strip().upper().replace(" ", "_")
def norm_adr(s):  return str(s).strip().lower().replace(" ", "_")

def load_assets():
    emb = np.load(EMB)
    nodes = pd.read_csv(NODES)
    node_to_idx = {n:i for i,n in enumerate(nodes["Node"])}
    meta = np.load(FEAT_COLS, allow_pickle=True).item()
    cols, mean, scale = meta["cols"], meta["mean"], meta["scale"]
    model = Scorer(len(cols))
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()
    cal = pickle.load(open(CAL, "rb"))
    return emb, node_to_idx, cols, mean, scale, model, cal

def get_vec(node_to_idx, emb, key, d):
    idx = node_to_idx.get(key)
    return emb[idx] if idx is not None else np.zeros(d, dtype=np.float32)

def build_row(d,g,a, emb, node_to_idx, base_dim):
    D, G, A = f"DRUG_{d}", (f"GENE_{g}" if g else None), f"ADR_{a}"
    vd = get_vec(node_to_idx, emb, D, base_dim)
    vg = get_vec(node_to_idx, emb, G, base_dim) if G else np.zeros_like(vd)
    va = get_vec(node_to_idx, emb, A, base_dim)

    had_dg = vd * vg; had_da = vd * va; had_ga = vg * va
    l1_dg = np.abs(vd - vg); l1_da = np.abs(vd - va); l1_ga = np.abs(vg - va)

    def cos(u,v):
        nu = np.linalg.norm(u)+1e-9; nv=np.linalg.norm(v)+1e-9
        return float(np.dot(u,v)/(nu*nv))
    cos_dg = cos(vd, vg) if vg.any() else 0.0
    cos_da = cos(vd, va); cos_ga = cos(vg, va) if vg.any() else 0.0

    feats = np.concatenate([vd, vg, va, had_dg, had_da, had_ga, l1_dg, l1_da, l1_ga]).astype(np.float32)
    extra = np.array([cos_dg, cos_da, cos_ga, 6,6,6, 0,0,0,0,0,0], dtype=np.float32)  # no KG+PubMed at inference here
    return np.concatenate([feats, extra])

def score(drug, gene, adr):
    d, g, a = norm_drug(drug), norm_gene(gene) if gene else "", norm_adr(adr)
    emb, node_to_idx, cols, mean, scale, model, cal = load_assets()
    base_dim = emb.shape[1]  # 128
    x = build_row(d,g,a, emb, node_to_idx, base_dim)
    # Align to training columns length
    if x.shape[0] != len(cols):
        # pad or trim
        if x.shape[0] < len(cols): 
            x = np.pad(x, (0, len(cols)-x.shape[0]))
        else:
            x = x[:len(cols)]
    x_std = (x - mean) / (scale + 1e-9)
    with torch.no_grad():
        p = float(model(torch.tensor(x_std[None,:], dtype=torch.float32)).numpy().ravel()[0])
    p_cal = float(cal.predict([p])[0])
    return p_cal

if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    g = sys.argv[2] if len(sys.argv)>3 else ""
    a = sys.argv[3] if len(sys.argv)>3 else sys.argv[2]
    print(f"Calibrated risk({d},{g},{a}) =", score(d,g,a))
