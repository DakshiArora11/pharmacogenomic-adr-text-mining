import pandas as pd
import numpy as np
import argparse, os, re, random
from collections import defaultdict, Counter

p = argparse.ArgumentParser()
p.add_argument("--triples", default="data/triples_normalized.csv")
p.add_argument("--kg_edges", default="data/kg_edges.csv")
p.add_argument("--out", default="output/triples_hardneg.csv")
p.add_argument("--neg_per_pos", type=int, default=2)  # how many hard negs per positive
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed)

# ---------- Helpers ----------
def norm_drug(s): return str(s).strip().lower().replace(" ", "_")
def norm_gene(s): return str(s).strip().upper().replace(" ", "_")
def norm_adr(s):  return str(s).strip().lower().replace(" ", "_")

print("📥 Loading triples & KG...")
df = pd.read_csv(args.triples)
df = df.dropna(subset=["Drug","ADR"]).copy()
df["drug"] = df["Drug"].map(norm_drug)
df["gene"] = df["Gene"].fillna("").map(norm_gene)
df["adr"]  = df["ADR"].map(norm_adr)

# Known positives set
pos_set = set(zip(df["drug"], df["gene"], df["adr"]))

# Neighborhoods from KG (same-type near nodes become candidate replacements)
kg = pd.read_csv(args.kg_edges)
nodes_by_type = defaultdict(set)
for n in pd.read_csv("data/dga_nodes_expanded.csv")["Node"].tolist():
    if n.startswith("DRUG_"): nodes_by_type["drug"].add(n[5:])
    elif n.startswith("GENE_"): nodes_by_type["gene"].add(n[5:])
    elif n.startswith("ADR_"):  nodes_by_type["adr"].add(n[4:])

# Build “neighbor” lists by relation for replacement sampling
print("🔎 Building replacement pools...")
drug_like   = sorted(nodes_by_type["drug"])
gene_like   = sorted(nodes_by_type["gene"])
adr_like    = sorted(nodes_by_type["adr"])

def sample_similar(item, pool, k=25):
    # simple: uniform sample from the same type; can be replaced by similarity-aware sampling later
    if item in pool:
        pool_ = [x for x in pool if x != item]
    else:
        pool_ = pool
    if not pool_: return []
    idx = np.random.choice(len(pool_), size=min(k, len(pool_)), replace=False)
    return [pool_[i] for i in idx]

rows = []
for d,g,a in pos_set:
    rows.append({"drug": d, "gene": g, "adr": a, "label": 1, "neg_type": "pos"})

print("🧪 Creating hard negatives...")
added = 0
for d,g,a in pos_set:
    candidates = []
    # corrupt gene
    if g:
        for g2 in sample_similar(g, gene_like, k=args.neg_per_pos):
            candidates.append((d,g2,a,"neg_gene"))
    # corrupt adr
    for a2 in sample_similar(a, adr_like, k=args.neg_per_pos):
        candidates.append((d,g,a2,"neg_adr"))
    # corrupt drug
    for d2 in sample_similar(d, drug_like, k=args.neg_per_pos):
        candidates.append((d2,g,a,"neg_drug"))

    # add if not positive elsewhere
    for D,G,A,kind in candidates:
        if (D,G,A) not in pos_set:
            rows.append({"drug": D, "gene": G, "adr": A, "label": 0, "neg_type": kind})
            added += 1

out = pd.DataFrame(rows).drop_duplicates()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
out.to_csv(args.out, index=False)
print(f"💾 Saved → {args.out}")
print("📊 Counts:", Counter(out["label"]))
print("🧾 Sample:", out.head(6))
