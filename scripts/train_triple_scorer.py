import torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from collections import Counter
from tqdm import tqdm

INP = "output/features_rich.csv"
MODEL_OUT = "models/triple_scorer.pt"
SCALER_OUT = "models/triple_scorer_scaler.npy"

EPOCHS = 40
LR = 1e-3
BATCH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("📥 Loading features...")
df = pd.read_csv(INP)
y = df["label"].values.astype(np.float32)
num_cols = [c for c in df.columns if c.startswith("f") or c.startswith("cos_") or c.startswith("sp_") or c.startswith("lit_")]
X = df[num_cols].values.astype(np.float32)

# split by drug
drugs = df["drug"].unique()
np.random.shuffle(drugs)
split = int(0.8 * len(drugs))
train_mask = df["drug"].isin(drugs[:split])
test_mask  = df["drug"].isin(drugs[split:])

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"✅ Train: {len(y_train)}  Test: {len(y_test)}  Labels: {Counter(y_train)}")

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
np.save(SCALER_OUT, {"mean": scaler.mean_, "scale": scaler.scale_, "cols": num_cols}, allow_pickle=True)

# datasets
Xt = torch.tensor(X_train, dtype=torch.float32)
Yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_test, dtype=torch.float32)
Yv = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, Yt), batch_size=BATCH, shuffle=True)

# model
class Scorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

model = Scorer(X_train.shape[1]).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

print("🧠 Training triple scorer...")
best_auc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    tot = 0
    for xb, yb in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        p = model(xb)
        loss = criterion(p, yb)
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(xb)
    tr_loss = tot / len(loader.dataset)

    model.eval()
    with torch.no_grad():
        pv = model(Xv.to(DEVICE)).cpu().numpy().ravel()
    auc = roc_auc_score(y_test, pv)
    scheduler.step(1 - auc)
    print(f"Epoch {epoch:03d} | Loss {tr_loss:.4f} | AUROC {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_OUT)

print(f"💾 Saved best → {MODEL_OUT}")

# final metrics
model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
model.eval()
with torch.no_grad():
    pv = model(Xv.to(DEVICE)).cpu().numpy().ravel()
pred = (pv > 0.5).astype(int)

auroc = roc_auc_score(y_test, pv)
auprc = average_precision_score(y_test, pv)
f1 = f1_score(y_test, pred)
mcc = matthews_corrcoef(y_test, pred)
print("\n📊 Final:")
print(f"   AUROC: {auroc:.4f}")
print(f"   AUPRC: {auprc:.4f}")
print(f"   F1:    {f1:.4f}")
print(f"   MCC:   {mcc:.4f}")
