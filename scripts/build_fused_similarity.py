import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1️⃣ Load input similarity matrices
# ============================================================
print("🔹 Loading similarity matrices...")

mesh_sim = pd.read_csv("data/adr_mesh_similarity.csv", index_col=0)
gda_sim = pd.read_csv("data/adr_gda_similarity.csv", index_col=0)

print(f"✅ Loaded MeSH sim: {mesh_sim.shape}, non-zero: {(mesh_sim.values != 0).sum()}")
print(f"✅ Loaded GDA sim:  {gda_sim.shape}, non-zero: {(gda_sim.values != 0).sum()}")

# Ensure the same order of ADR IDs
common = sorted(set(mesh_sim.index) & set(gda_sim.index))
mesh_sim = mesh_sim.loc[common, common]
gda_sim = gda_sim.loc[common, common]

# ============================================================
# 2️⃣ Normalize each similarity matrix
# ============================================================
print("🔹 Normalizing matrices to [0, 1] range...")

def normalize_matrix(df):
    scaler = MinMaxScaler()
    arr = df.values
    if np.max(arr) == np.min(arr):  # avoid div by zero
        return df
    arr_scaled = scaler.fit_transform(arr)
    return pd.DataFrame(arr_scaled, index=df.index, columns=df.columns)

mesh_norm = normalize_matrix(mesh_sim)
gda_norm = normalize_matrix(gda_sim)

# ============================================================
# 3️⃣ Fuse matrices (weighted sum)
# ============================================================
alpha = 0.5  # weight for GDA; adjust as desired (0.5 = equal weight)
fused = alpha * gda_norm + (1 - alpha) * mesh_norm

non_zero = np.count_nonzero(fused.values)
print(f"✅ Fused matrix created with shape {fused.shape}, non-zero entries: {non_zero}")

# ============================================================
# 4️⃣ Save fused matrix
# ============================================================
out_path = "data/adr_fused_similarity.csv"
fused.to_csv(out_path)
print(f"💾 Saved fused similarity matrix → {out_path}")
