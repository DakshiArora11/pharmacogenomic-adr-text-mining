import joblib
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import shap

print("📥 Loading model + PCA + scaler...")
model = joblib.load("models/fusion_model_rich.pkl")
scaler = joblib.load("models/fusion_scaler.pkl")
pca = joblib.load("models/fusion_pca.pkl")

# Load dataset for feature names
df = pd.read_csv("D:/dganet_features/features_rich_full.csv.gz", compression="gzip")

# Drop non-numeric columns
df = df.drop(columns=df.select_dtypes(exclude=['number']).columns)
if "label" in df.columns:
    y = df["label"].astype(int)
    df = df.drop(columns=["label"])

print(f"✅ Features loaded: {df.shape}")

# Align PCA-transformed space
Xp = scaler.transform(df)
Xp = pca.transform(Xp)

# --- LightGBM Feature Importances ---
plt.figure(figsize=(8, 10))
lgb.plot_importance(model, max_num_features=30, importance_type="gain")
plt.title("Top 30 Feature Importances (Post-PCA)")
plt.tight_layout()
plt.show()

# --- SHAP Interpretability (Optional but insightful) ---
print("⚙️ Running SHAP analysis (sampling 5000 rows)...")
explainer = shap.Explainer(model)
sample = Xp[:5000]
shap_values = explainer(sample)

shap.summary_plot(shap_values, sample, show=True)
