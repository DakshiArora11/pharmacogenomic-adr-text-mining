import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse

p = argparse.ArgumentParser()
p.add_argument('--features_csv', default='output/features.csv')
args = p.parse_args()

df = pd.read_csv(args.features_csv)

# take only the numeric features
X = df.iloc[:,3:].values

# reduce to 2 dimensions
pca = PCA(n_components=2)
xy = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(xy[:,0], xy[:,1], alpha=0.6)

# annotate each point with drug-gene-adr
labels = (df['drug'].fillna('') + '-' +
          df['gene'].fillna('') + '-' +
          df['adr'].fillna('')).tolist()

for i, txt in enumerate(labels):
    plt.annotate(txt, (xy[i,0], xy[i,1]), fontsize=6, alpha=0.7)

plt.title('2D PCA of triple features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()
