import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import defaultdict


def evaluate_embeddings(X_emb, nodes, edges, subset=None):
    """Train-test evaluation on link prediction."""
    node2idx = {n: i for i, n in enumerate(nodes["Node"])}

    # Filter edges for specific subset (drug/gene/ADR)
    if subset:
        if subset == "drug":
            mask = edges["Source"].str.startswith("DRUG_")
        elif subset == "gene":
            mask = edges["Source"].str.startswith("GENE_")
        elif subset == "adr":
            mask = edges["Target"].str.startswith("ADR_")
        else:
            mask = np.ones(len(edges), dtype=bool)
        edges = edges[mask]

    # Positive pairs
    X_pos = []
    for _, row in edges.iterrows():
        if row["Source"] in node2idx and row["Target"] in node2idx:
            i, j = node2idx[row["Source"]], node2idx[row["Target"]]
            X_pos.append(np.abs(X_emb[i] - X_emb[j]))
    if not X_pos:
        return None
    X_pos = np.array(X_pos)
    y_pos = np.ones(len(X_pos))

    # Negative pairs
    rng = np.random.default_rng(42)
    drugs = [n for n in nodes["Node"] if n.startswith("DRUG_")]
    adrs = [n for n in nodes["Node"] if n.startswith("ADR_")]
    genes = [n for n in nodes["Node"] if n.startswith("GENE_")]

    X_neg = []
    for _ in range(len(X_pos)):
        if subset == "drug":
            d, a = rng.choice(drugs), rng.choice(adrs)
        elif subset == "gene":
            d, a = rng.choice(genes), rng.choice(adrs)
        else:
            d, a = rng.choice(drugs), rng.choice(adrs)
        if not ((edges["Source"] == d) & (edges["Target"] == a)).any():
            i, j = node2idx[d], node2idx[a]
            X_neg.append(np.abs(X_emb[i] - X_emb[j]))
    X_neg = np.array(X_neg)
    y_neg = np.zeros(len(X_neg))

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([y_pos, y_neg])

    # Split & train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]

    return {
        "AUROC": roc_auc_score(y_test, y_pred),
        "AUPRC": average_precision_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred > 0.5),
        "MCC": matthews_corrcoef(y_test, y_pred > 0.5),
    }


def print_metrics(name, metrics):
    if metrics:
        print(f"{name:<10} | AUROC={metrics['AUROC']:.4f} | AUPRC={metrics['AUPRC']:.4f} | "
              f"F1={metrics['F1']:.4f} | MCC={metrics['MCC']:.4f}")
    else:
        print(f"{name:<10} | No valid edges to evaluate.")


print("🔹 Loading embeddings and edges...")

base_emb = np.load("data/dga_embeddings.npy")
lit_emb = np.load("data/dganet_literature_embeddings.npy")
nodes = pd.read_csv("data/dga_nodes.csv")
edges = pd.read_csv("data/dga_edges.csv")
edges = edges[edges["Relation"] == "has_side_effect"]

categories = ["all", "drug", "gene", "adr"]
results = defaultdict(dict)

for cat in categories:
    print(f"\n🔹 Evaluating category: {cat}")
    base_metrics = evaluate_embeddings(base_emb, nodes, edges, subset=cat)
    lit_metrics = evaluate_embeddings(lit_emb, nodes, edges, subset=cat)
    results[cat]["baseline"] = base_metrics
    results[cat]["literature"] = lit_metrics

print("\n📊 DGAnet Performance Comparison by Category")
print("------------------------------------------------------------")
print("Category   | Model       | AUROC   | AUPRC  | F1     | MCC")
print("------------------------------------------------------------")
for cat in categories:
    for model in ["baseline", "literature"]:
        m = results[cat][model]
        if m:
            print(f"{cat:<10} | {model:<11} | {m['AUROC']:.4f} | {m['AUPRC']:.4f} | {m['F1']:.4f} | {m['MCC']:.4f}")
print("------------------------------------------------------------")
