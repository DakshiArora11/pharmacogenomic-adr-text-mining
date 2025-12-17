import pandas as pd
import os

# --- Inputs ---
triples_path = "data/triples_normalized.csv"
nodes_path   = "data/dga_nodes_expanded.csv"
out_path     = "data/kg_edges.csv"

print("📥 Loading data...")
triples = pd.read_csv(triples_path)
nodes   = pd.read_csv(nodes_path)

# Normalize columns
triples.columns = [c.strip().lower() for c in triples.columns]
existing_nodes = set(nodes["Node"])

def d_key(d): return f"DRUG_{d.lower().replace(' ', '_')}"
def g_key(g): return f"GENE_{g.upper().replace(' ', '_')}"
def a_key(a): return f"ADR_{a.replace(' ', '_')}"

edges = []

# --- Build edges ---
for _, r in triples.iterrows():
    d, g, a = r.get("drug"), r.get("gene"), r.get("adr")

    if pd.notna(d):
        dnode = d_key(d)
        if pd.notna(g):
            gnode = g_key(g)
            edges.append((dnode, "interacts_with", gnode))
        if pd.notna(a):
            anode = a_key(a)
            edges.append((dnode, "causes", anode))
    if pd.notna(g) and pd.notna(a):
        edges.append((g_key(g), "associated_with", a_key(a)))

# --- Create DataFrame ---
df_edges = pd.DataFrame(edges, columns=["src", "rel", "dst"]).drop_duplicates().reset_index(drop=True)

# --- Filter to valid nodes ---
valid_edges = df_edges[df_edges["src"].isin(existing_nodes) & df_edges["dst"].isin(existing_nodes)]

print(f"🔗 Built {len(df_edges):,} raw edges → {len(valid_edges):,} valid edges retained")
print("🧩 Example edges:")
print(valid_edges.sample(10, random_state=42))

# --- Save ---
os.makedirs(os.path.dirname(out_path), exist_ok=True)
valid_edges.to_csv(out_path, index=False)
print(f"💾 Saved KG edges to {out_path}")
