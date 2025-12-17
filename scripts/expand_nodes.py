import pandas as pd
import os

# --- Inputs ---
triples_path = "data/triples_normalized.csv"
nodes_path   = "data/dga_nodes.csv"
out_path     = "data/dga_nodes_expanded.csv"

print("📥 Loading data...")
triples = pd.read_csv(triples_path)
nodes   = pd.read_csv(nodes_path)

# --- Normalize triples ---
triples.columns = [c.strip().lower() for c in triples.columns]
drugs = sorted(triples["drug"].dropna().unique().tolist())
genes = sorted(triples["gene"].dropna().unique().tolist())
adrs  = sorted(triples["adr"].dropna().unique().tolist())

existing = set(nodes["Node"])

# --- Build new node names ---
def drug_key(d): return f"DRUG_{d.lower().replace(' ','_')}"
def gene_key(g): return f"GENE_{g.upper().replace(' ','_')}"
def adr_key(a):  return f"ADR_{a.replace(' ','_')}"   # keep text form for now

new_drugs = [drug_key(d) for d in drugs if drug_key(d) not in existing]
new_genes = [gene_key(g) for g in genes if gene_key(g) not in existing]
new_adrs  = [adr_key(a)  for a in adrs  if adr_key(a)  not in existing]

print(f"➕ Missing: {len(new_drugs)} drugs, {len(new_genes)} genes, {len(new_adrs)} adrs")

# --- Concatenate & deduplicate ---
expanded = pd.concat([
    nodes,
    pd.DataFrame({"Node": new_drugs + new_genes + new_adrs})
], ignore_index=True).drop_duplicates().reset_index(drop=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
expanded.to_csv(out_path, index=False)

print(f"💾 Saved expanded node list → {out_path}")
print(f"📊 Old: {len(nodes)} | New: {len(expanded)} (added {len(expanded)-len(nodes)})")
print("✅ Example:")
print(expanded.tail(10))
