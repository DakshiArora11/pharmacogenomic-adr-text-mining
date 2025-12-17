import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import pickle, os, random
from collections import Counter

print("📥 Loading fused feature table...")
df = pd.read_csv("output/features_fused.csv")

# --- Step 1: Filter out zero embeddings ---
non_zero_mask = (df.filter(like="f").sum(axis=1) != 0)
removed = len(df) - non_zero_mask.sum()
df = df[non_zero_mask]
print(f"⚙️ Filtered out {removed:,} rows with zero embeddings → {len(df):,} remaining")

# --- Step 2: Add balanced negatives ---
if "label" not in df.columns or df["label"].nunique() == 1:
    print("🧪 Generating balanced hard negatives...")
    pos_df = df.copy()
    pos_df["label"] = 1
    all_adrs = df["adr"].dropna().unique().tolist()
    neg_rows = []

    for drug in pos_df["drug"].dropna().unique():
        ddf = pos_df[pos_df["drug"] == drug]
        seen_adrs = set(ddf["adr"].dropna())
        genes = ddf["gene"].dropna().unique().tolist()

        for _ in range(max(5, len(seen_adrs))):
            neg_adr = random.choice(all_adrs)
            if neg_adr not in seen_adrs:
                neg_rows.append({
                    "drug": drug,
                    "gene": random.choice(genes) if genes else "unknown",
                    "adr": neg_adr,
                    "mesh_id": "",
                    "label": 0
                })

    neg_df = pd.DataFrame(neg_rows)
    # Force 1:1 ratio
    neg_df = neg_df.sample(n=len(pos_df), replace=True, random_state=42)
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    print(f"✅ Balanced dataset: {df['label'].value_counts().to_dict()}")

# --- Step 3: PubMed evidence ---
if os.path.exists("data/pubmed_results_large.csv"):
    print("🔗 Integrating PubMed evidence...")
    pubmed = pd.read_csv("data/pubmed_results_large.csv")
    pub_hits = {}
    abstracts = pubmed["Abstract"].astype(str).str.lower().tolist()
    for a in abstracts:
        for d in df["drug"].dropna().unique():
            dl = d.lower()
            if dl in a:
                for adr in df["adr"].dropna().unique():
                    al = adr.lower().replace("_", " ")
                    if al in a:
                        pub_hits[(d, adr)] = pub_hits.get((d, adr), 0) + 1
    df["pubmed_hits"] = df.apply(lambda r: pub_hits.get((r["drug"], r["adr"]), 0), axis=1)
else:
    print("⚠️ PubMed file not found — skipping evidence integration.")
    df["pubmed_hits"] = 0

# --- Step 4: Clean ---
text_cols = ["drug", "gene", "adr", "mesh_id"]
df[text_cols] = df[text_cols].fillna("unknown")
df["pubmed_hits"] = df["pubmed_hits"].fillna(0)

num_cols = df.filter(like="f").columns
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
print("✅ Cleaned dataset — no NaNs or Infs remain.")

# --- Step 5: Split by unseen drugs ---
unique_drugs = df["drug"].unique().tolist()
random.shuffle(unique_drugs)
split_idx = int(len(unique_drugs) * 0.8)
train_drugs, val_drugs = unique_drugs[:split_idx], unique_drugs[split_idx:]

train_mask = df["drug"].isin(train_drugs)
test_mask = df["drug"].isin(val_drugs)

X_train = df.loc[train_mask].filter(like="f").values
y_train = df.loc[train_mask, "label"].values
X_test  = df.loc[test_mask].filter(like="f").values
y_test  = df.loc[test_mask, "label"].values

X_train = np.hstack([X_train, df.loc[train_mask, ["pubmed_hits"]].values])
X_test  = np.hstack([X_test, df.loc[test_mask, ["pubmed_hits"]].values])

# --- Sanity ---
assert not np.isnan(X_train).any()
assert not np.isnan(X_test).any()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(n_components=min(100, X_train.shape[1]), random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(f"📉 Reduced feature dimension to {X_train.shape[1]} using PCA")
print(f"✅ Training samples: {len(y_train):,} | Validation: {len(y_test):,}")
print(f"🧾 Label balance: {Counter(y_train)} (train), {Counter(y_test)} (test)")

# --- Step 6: Train ---
clf = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_child_samples=20,
    random_state=42
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

# --- Step 7: Evaluate ---
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

# --- Step 8: Save ---
os.makedirs("models", exist_ok=True)
with open("models/fusion_model_fixed.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n💾 Saved retrained model → models/fusion_model_fixed.pkl")
