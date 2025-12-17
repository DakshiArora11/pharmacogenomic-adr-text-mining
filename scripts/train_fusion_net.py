import os
import random
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tqdm import tqdm
import joblib

# ------------------------- Config -------------------------
FEATURES_CSV = "output/features_fused.csv"
PUBMED_CSV   = "data/pubmed_results_large.csv"   # optional; adds evidence feature if present
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "fusion_net.pth")
META_PATH    = os.path.join(MODEL_DIR, "fusion_net_meta.pkl")

SEED = 42
VAL_DRUG_SPLIT = 0.2
USE_PCA = True
PCA_COMPONENTS = 100
BATCH_SIZE = 512
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- Utils -------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        AUROC = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        AUPRC = average_precision_score(y_true, y_prob),
        F1    = f1_score(y_true, y_pred),
        MCC   = matthews_corrcoef(y_true, y_pred)
    )

def youden_threshold(y_true, y_prob):
    # simple threshold sweep to maximize Youden's J
    probs = np.asarray(y_prob)
    thr_space = np.linspace(0.05, 0.95, 19)
    best_thr, best_j = 0.5, -1
    from sklearn.metrics import confusion_matrix
    for t in thr_space:
        y_pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_thr = j, t
    return best_thr

# ------------------------- Data Prep -------------------------
def load_and_prepare():
    print("📥 Loading fused feature table...")
    df = pd.read_csv(FEATURES_CSV)

    # drop pure-zero rows (rare after your alignment, but safe)
    nz_mask = (df.filter(like="f").sum(axis=1) != 0)
    dropped = len(df) - nz_mask.sum()
    df = df[nz_mask].reset_index(drop=True)
    print(f"⚙️ Filtered out {dropped} zero-feature rows → {len(df)} remaining")

    # Add PubMed evidence as scalar feature if available
    if os.path.exists(PUBMED_CSV):
        print("🔗 Integrating PubMed evidence…")
        pubmed = pd.read_csv(PUBMED_CSV)
        hits = Counter()
        # very lightweight co-occurrence (prototype-friendly)
        abstracts = pubmed["Abstract"].astype(str).str.lower().tolist()
        uniq_drugs = df["drug"].unique().tolist()
        uniq_adrs  = df["adr"].unique().tolist()
        for a in tqdm(abstracts, desc="Scanning abstracts", total=len(abstracts)):
            for d in uniq_drugs:
                dl = d.lower()
                if dl in a:
                    for adr in uniq_adrs:
                        al = adr.lower().replace("_"," ")
                        if al in a:
                            hits[(d, adr)] += 1
        df["pubmed_hits"] = df.apply(lambda r: hits.get((r["drug"], r["adr"]), 0), axis=1)
    else:
        print("⚠️ PubMed file not found — skipping evidence integration.")
        df["pubmed_hits"] = 0

    # Hard negatives (balanced 1:1) if label missing or single-class
    if "label" not in df.columns or df["label"].nunique() == 1:
        print("🧪 Generating balanced hard negatives…")
        pos_df = df.copy()
        pos_df["label"] = 1
        all_adrs = df["adr"].unique().tolist()
        neg_rows = []
        for drug in tqdm(pos_df["drug"].unique(), desc="Negatives per drug"):
            ddf = pos_df[pos_df["drug"] == drug]
            seen_adrs = set(ddf["adr"])
            genes = ddf["gene"].unique().tolist()
            # try to generate a reasonable number of in-context negatives per drug
            target_neg = max(10, len(seen_adrs))
            tries = 0
            while len(neg_rows) < len(pos_df) and tries < target_neg * 5:
                tries += 1
                neg_adr = random.choice(all_adrs)
                if neg_adr not in seen_adrs:
                    neg_rows.append(dict(
                        drug=drug,
                        gene=random.choice(genes),
                        adr=neg_adr,
                        mesh_id="",
                        pubmed_hits=0,  # conservative; you can compute co-occur for neg too if desired
                        label=0
                    ))
        neg_df = pd.DataFrame(neg_rows)
        # Balance 1:1
        if len(neg_df) < len(pos_df):
            pos_df = pos_df.sample(n=len(neg_df), random_state=SEED)
        else:
            neg_df = neg_df.sample(n=len(pos_df), random_state=SEED)
        df = pd.concat([pos_df, neg_df], ignore_index=True)
        print(f"✅ Balanced dataset: {df['label'].value_counts().to_dict()}")
    else:
        print("✅ Using provided labels")

    return df

def split_by_drug(df):
    drugs = df["drug"].unique().tolist()
    random.shuffle(drugs)
    cut = int(len(drugs) * (1 - VAL_DRUG_SPLIT))
    train_drugs, val_drugs = drugs[:cut], drugs[cut:]
    tr = df[df["drug"].isin(train_drugs)].reset_index(drop=True)
    va = df[df["drug"].isin(val_drugs)].reset_index(drop=True)
    print(f"🧪 Split by unseen drugs → Train: {len(tr):,}  Val: {len(va):,}")
    return tr, va

def build_matrices(df_tr, df_va):
    # features = f* columns + pubmed_hits
    feat_cols = [c for c in df_tr.columns if c.startswith("f")]
    add_cols  = ["pubmed_hits"]
    X_tr = df_tr[feat_cols + add_cols].values
    y_tr = df_tr["label"].values.astype(np.float32)
    X_va = df_va[feat_cols + add_cols].values
    y_va = df_va["label"].values.astype(np.float32)

    # scale + optional PCA
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    pca = None
    if USE_PCA:
        pca = PCA(n_components=min(PCA_COMPONENTS, X_tr.shape[1]), random_state=SEED)
        X_tr = pca.fit_transform(X_tr)
        X_va = pca.transform(X_va)
        print(f"📉 PCA reduced dims → {X_tr.shape[1]}")

    return X_tr, y_tr, X_va, y_va, scaler, pca

# ------------------------- Model -------------------------
class FusionNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # logits
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(1, n)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    thr = youden_threshold(ys, probs)
    m = metrics(ys, probs, thr=thr)
    m["thr"] = float(thr)
    return m

# ------------------------- Main -------------------------
def main():
    seed_everything(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_prepare()
    df_tr, df_va = split_by_drug(df)
    X_tr, y_tr, X_va, y_va, scaler, pca = build_matrices(df_tr, df_va)

    # tensors & loaders
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)

    tr_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    va_ds = torch.utils.data.TensorDataset(X_va_t, y_va_t)
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    va_ld = torch.utils.data.DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = FusionNet(in_dim=X_tr.shape[1]).to(DEVICE)

    # class weight to handle any residual imbalance
    pos_ratio = y_tr.mean()
    w_pos = 0.5 / (pos_ratio + 1e-9)
    w_neg = 0.5 / (1 - pos_ratio + 1e-9)
    def weighted_bce_with_logits():
        # logits, y in {0,1}
        def loss_fn(logits, y):
            weight = torch.where(y > 0.5, torch.tensor(w_pos, device=logits.device), torch.tensor(w_neg, device=logits.device))
            return nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weight)
        return loss_fn

    criterion = weighted_bce_with_logits()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auc = -1
    best_state = None
    patience = PATIENCE

    print(f"🚀 Training on {DEVICE} for {EPOCHS} epochs…")
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, tr_ld, criterion, optimizer)
        m = eval_epoch(model, va_ld)
        print(f"Epoch {epoch:02d} | loss {tr_loss:.4f} | "
              f"AUROC {m['AUROC']:.3f} AUPRC {m['AUPRC']:.3f} F1 {m['F1']:.3f} MCC {m['MCC']:.3f} thr {m['thr']:.2f}")

        if m["AUROC"] > best_auc + 1e-4:
            best_auc = m["AUROC"]
            best_state = {
                "model": model.state_dict(),
                "meta": dict(
                    scaler=scaler,
                    pca=pca,
                    in_dim=X_tr.shape[1],
                    thr=m["thr"]
                )
            }
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                print("⏹️ Early stopping.")
                break

    # save best
    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "meta": dict(scaler=scaler, pca=pca, in_dim=X_tr.shape[1], thr=0.5)
        }

    torch.save(best_state["model"], MODEL_PATH)
    # serialize scaler & pca with joblib (pickle-safe)
    joblib.dump(dict(scaler=scaler, pca=pca, in_dim=X_tr.shape[1], thr=best_state["meta"]["thr"]), META_PATH)

    print(f"💾 Saved model → {MODEL_PATH}")
    print(f"💾 Saved preprocessing → {META_PATH}")

    # final eval with best state
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    final = eval_epoch(model, va_ld)
    print("\n📊 Final (unseen drugs) validation:")
    for k, v in final.items():
        if k != "thr":
            print(f"  {k:>6}: {v:.4f}")
    print(f"  thr  : {final['thr']:.2f}")

if __name__ == "__main__":
    main()
