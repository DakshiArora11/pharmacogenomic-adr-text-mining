import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import joblib
import os

# --- Config ---
features_csv = "output/features_fused.csv"        # built via assemble_features later
model_out = "models/fusion_model.pkl"
os.makedirs("models", exist_ok=True)

print("📥 Loading feature table...")
df = pd.read_csv(features_csv)
print("✅ Loaded:", df.shape)

# --- Separate features & labels ---
X = df[[c for c in df.columns if c.startswith("f")]].values
y = df["label"].values

# --- Train/validation split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"🧠 Training samples: {X_train.shape[0]} | Validation samples: {X_test.shape[0]}")

# --- Train LightGBM ---
params = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
)

clf = lgb.LGBMClassifier(**params, n_estimators=500)
clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.log_evaluation(period=50)]  # replaces 'verbose'
)

# --- Evaluate ---
y_pred_prob = clf.predict_proba(X_test)[:, 1]
y_pred_label = (y_pred_prob > 0.5).astype(int)

metrics = {
    "AUROC": roc_auc_score(y_test, y_pred_prob),
    "AUPRC": average_precision_score(y_test, y_pred_prob),
    "F1": f1_score(y_test, y_pred_label),
    "MCC": matthews_corrcoef(y_test, y_pred_label),
}

print("📊 Model Performance:")
for k, v in metrics.items():
    print(f"{k:>8}: {v:.4f}")

# --- Save model ---
joblib.dump(clf, model_out)
print(f"💾 Saved trained model → {model_out}")
