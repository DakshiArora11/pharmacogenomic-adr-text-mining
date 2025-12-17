import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load ADR similarity matrix ---
sim_path = "data/adr_mesh_similarity.csv"   # update if needed
df = pd.read_csv(sim_path, index_col=0)

print("Loaded ADR-ADR similarity matrix:", df.shape)

# --- Optional: Limit to top N ADRs for readable visualization ---
N = 50
df_small = df.iloc[:N, :N]

# --- Plot ---
plt.figure(figsize=(14, 10))
sns.heatmap(
    df_small,
    cmap="viridis",
    linewidths=0.1,
    linecolor="gray",
    square=True,
    cbar_kws={"label": "Similarity Score"},
)

plt.title("ADR–ADR Similarity Heatmap (MeSH Ontology-based)", fontsize=16)
plt.xlabel("ADRs")
plt.ylabel("ADRs")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
