import numpy as np, pandas as pd, pickle, os, torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

FEATS = "output/features_rich.csv"
MODEL = "models/triple_scorer.pt"
SCALER = "models/triple_scorer_scaler.npy"
CAL_OUT = "models/calibrator_isotonic.pkl"

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

print("📥 Loading...")
df = pd.read_csv(FEATS)
cols = np.load(SCALER, allow_pickle=True).item()["cols"]
mean = np.load(SCALER, allow_pickle=True).item()["mean"]
scale = np.load(SCALER, allow_pickle=True).item()["scale"]

X = df[cols].values.astype(np.float32)
X = (X - mean) / (scale + 1e-9)
y = df["label"].values.astype(np.float32)

# 20% for calibration
_, X_cal, _, y_cal = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Scorer(X.shape[1])
model.load_state_dict(torch.load(MODEL, map_location="cpu"))
model.eval()
with torch.no_grad():
    p = model(torch.tensor(X_cal)).numpy().ravel()

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p, y_cal)
pickle.dump(iso, open(CAL_OUT, "wb"))
print(f"💾 Saved calibrator → {CAL_OUT}")
