import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/adr_gda_similarity.csv", index_col=0)
df_small = df.iloc[:50, :50]

plt.figure(figsize=(16, 12))
sns.heatmap(df_small, cmap="plasma", square=True, cbar_kws={"label": "Gene–Gene Similarity"})
plt.title("Gene–Gene Similarity (GDA-based)", fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
