import pandas as pd, numpy as np, lightgbm as lgb, pickle, os, random
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.decomposition import PCA
from collections import Counter

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print("📥 Loading fused feature table...")
df = pd.read_csv("output/features_fused_strict.csv")
feature_cols = [c for c in df.columns if c.startswith("f")]

# --- Hard Negative Sampling ---
print("🧪 Generating hard negatives...")
pos = df.copy()
pos["label"] = 1

neg = pos.copy()
neg["gene"] = np.random.permutation(neg["gene"].values)
neg["adr"]  = np.random.permutation(neg["adr"].values)
neg["label"] = 0

df = pd.concat([pos, neg], ignore_index=True).sample(frac=1, random_state=SEED)
print(f"✅ Balanced dataset: {Counter(df['label'])}")

# --- Add Gaussian Noise ---
def add_noise(X, level=0.01):
    return X + np.random.normal(0, level, X.shape)

X = add_noise(df[feature_cols].values)
y = df["label"].values

# --- PCA Dimensionality Reduction ---
pca = PCA(n_components=100, random_state=SEED)
X = pca.fit_transform(X)

# --- Split by Unseen Drugs ---
unique_drugs = df["drug"].unique().tolist()
random.shuffle(unique_drugs)
split_idx = int(len(unique_drugs) * 0.85)
train_drugs, test_drugs = unique_drugs[:split_idx], unique_drugs[split_idx:]

train_mask = df["drug"].isin(train_drugs)
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"✅ Train: {len(y_train)} | Test: {len(y_test)}")

# --- Train LightGBM ---
clf = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=32,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50)]
)

# --- Evaluate ---
preds = clf.predict_proba(X_test)[:, 1]
y_pred = (preds > 0.5).astype(int)

auroc = roc_auc_score(y_test, preds)
auprc = average_precision_score(y_test, preds)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n📊 Model Performance (Unseen Drugs):")
print(f"   AUROC: {auroc:.4f}")
print(f"   AUPRC: {auprc:.4f}")
print(f"   F1:    {f1:.4f}")
print(f"   MCC:   {mcc:.4f}")

# --- Save Model ---
os.makedirs("models", exist_ok=True)
with open("models/fusion_model_strict.pkl", "wb") as f:
    pickle.dump(clf, f)

print("💾 Model saved → models/fusion_model_strict.pkl")
