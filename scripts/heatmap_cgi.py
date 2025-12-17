import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load matrix
df = pd.read_csv("data/drug_cgi_similarity.csv", index_col=0)

# Optional: sample top 40 drugs for cleaner visualization
df_small = df.iloc[:40, :40]

plt.figure(figsize=(14, 12))
sns.heatmap(df_small, cmap="viridis", linewidths=0.2)
plt.title("Drug–Drug Similarity Heatmap (CGI-based)")
plt.xlabel("Drugs")
plt.ylabel("Drugs")
plt.tight_layout()
plt.show()
