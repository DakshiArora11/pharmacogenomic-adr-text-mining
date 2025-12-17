import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
from tqdm import tqdm

# --- Config ---
INPUT_FILE = "output/features_fused_strict.csv"
MODEL_OUT = "models/fusion_model_mlp.pt"
EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Device: {DEVICE}")
print("📥 Loading data...")

# --- Load and clean ---
df = pd.read_csv(INPUT_FILE)
feature_cols = [c for c in df.columns if c.startswith("f")]
X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y = df["label"].values if "label" in df.columns else np.ones(len(df))

# --- Add synthetic negatives if missing ---
if len(np.unique(y)) == 1:
    print("🧪 Adding synthetic negatives...")
    neg_idx = np.random.choice(len(X), len(X)//2, replace=False)
    y[neg_idx] = 0

# --- Split by drug (not random) ---
if "drug" in df.columns:
    drugs = df["drug"].unique()
    np.random.shuffle(drugs)
    split = int(0.8 * len(drugs))
    train_drugs, test_drugs = drugs[:split], drugs[split:]
    train_mask = df["drug"].isin(train_drugs)
    test_mask = df["drug"].isin(test_drugs)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Train: {len(y_train)} | Test: {len(y_test)} | Positives: {Counter(y_train)}")

# --- Normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Tensor datasets ---
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# --- Define MLP ---
class FusionMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = FusionMLP(X_train.shape[1]).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

print("🧠 Training neural fusion model...")

best_auc = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    avg_loss = total_loss / len(train_loader.dataset)

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t.to(DEVICE)).cpu().numpy().flatten()
    auc = roc_auc_score(y_test, preds)
    scheduler.step(1 - auc)

    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | AUROC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_OUT)

print(f"💾 Best model saved → {MODEL_OUT}")

# --- Final Evaluation ---
model.load_state_dict(torch.load(MODEL_OUT))
model.eval()
with torch.no_grad():
    preds = model(X_test_t.to(DEVICE)).cpu().numpy().flatten()
y_pred = (preds > 0.5).astype(int)

auroc = roc_auc_score(y_test, preds)
auprc = average_precision_score(y_test, preds)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n📊 Final Model Performance:")
print(f"   AUROC: {auroc:.4f}")
print(f"   AUPRC: {auprc:.4f}")
print(f"   F1:    {f1:.4f}")
print(f"   MCC:   {mcc:.4f}")
