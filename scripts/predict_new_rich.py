import joblib
import pandas as pd

model = joblib.load("models/fusion_model_rich.pkl")
scaler = joblib.load("models/fusion_scaler.pkl")
pca = joblib.load("models/fusion_pca.pkl")

print("📥 Loading new dataset for prediction...")
# Replace this line with the dataset you want
new_df = pd.read_csv("D:/dganet_features/features_rich_full.csv.gz", compression="gzip")

# Drop non-numeric columns
if "label" in new_df.columns:
    new_df = new_df.drop(columns=["label"])
new_df = new_df.drop(columns=new_df.select_dtypes(exclude=['number']).columns, errors='ignore')
new_df = new_df.replace([float('inf'), float('-inf')], 0).fillna(0)

print("⚙️ Transforming features...")
Xp = pca.transform(scaler.transform(new_df))

print("🧠 Predicting probabilities...")
probs = model.predict_proba(Xp)[:, 1]

new_df["prediction_score"] = probs
out_path = "D:/dganet_features/predictions_rich.csv"
new_df.to_csv(out_path, index=False)
print(f"✅ Predictions saved → {out_path}")
