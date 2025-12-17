import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

print("📥 Loading rich feature table...")
df = pd.read_csv("D:/dganet_features/features_rich_full.csv.gz", compression="gzip")
print(f"✅ Loaded: {df.shape}")

# --- Identify and drop non-numeric columns ---
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"🧹 Dropping non-numeric columns: {non_numeric_cols[:10]}{' ...' if len(non_numeric_cols) > 10 else ''}")
X = df.drop(columns=non_numeric_cols)

# --- Detect label column ---
y = None
for possible_label in ["label", "y", "target", "is_positive"]:
    if possible_label in X.columns:
        y = X[possible_label].astype(int)
        X = X.drop(columns=[possible_label])
        print(f"✅ Found label column: '{possible_label}'")
        break

if y is None:
    print("⚠️ No label column found — generating synthetic balanced labels for unsupervised fusion check.")
    y = np.concatenate([np.ones(len(X)//2), np.zeros(len(X)//2)])
    X = X.iloc[: len(y), :]

# --- Clean ---
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# --- Scale + reduce ---
print("⚙️ Scaling and dimensionality reduction (PCA 256)...")
scaler = StandardScaler()
Xp = scaler.fit_transform(X)
pca = PCA(n_components=min(256, Xp.shape[1]), random_state=42)
Xp = pca.fit_transform(Xp)
print(f"✅ Reduced to shape: {Xp.shape}")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    Xp, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

# --- LightGBM params ---
params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "num_leaves": 64,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 2000,
    "n_jobs": -1,
    "verbose": -1
}

print("🧠 Training LightGBM on rich features...")

# Use callbacks for early stopping + logging
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100)
]

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=callbacks
)

# --- Evaluate ---
preds = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, preds)
auprc = average_precision_score(y_test, preds)
f1 = f1_score(y_test, (preds > 0.5).astype(int))
mcc = matthews_corrcoef(y_test, (preds > 0.5).astype(int))

print("\n📊 Model Performance (Rich Features):")
print(f"   AUROC: {auroc:.4f}")
print(f"   AUPRC: {auprc:.4f}")
print(f"   F1:    {f1:.4f}")
print(f"   MCC:   {mcc:.4f}")

# --- Save ---
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fusion_model_rich.pkl")
joblib.dump(scaler, "models/fusion_scaler.pkl")
joblib.dump(pca, "models/fusion_pca.pkl")

print("\n💾 Saved model, scaler, and PCA → models/")
print("✅ Done.")
