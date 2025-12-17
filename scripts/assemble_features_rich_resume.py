#!/usr/bin/env python3
"""
Memory-safe, resumable assembler for rich features.
Writes compressed CSV parts (gzip) to an output directory and updates a .progress file.
"""

import pandas as pd
import numpy as np
import argparse, os, shutil, json
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

# -----------------------
# CLI arguments
# -----------------------
p = argparse.ArgumentParser()
p.add_argument("--triples", default="output/triples_hardneg.csv")
p.add_argument("--emb", default="data/dganet_literature_embeddings_fused_aligned.npy")
p.add_argument("--nodes", default="data/dga_nodes_expanded.csv")
p.add_argument("--kg_edges", default="data/kg_edges.csv")
p.add_argument("--pubmed", default="data/pubmed_results_large.csv")  # or "none"
p.add_argument("--out_dir", default="output/features_rich_parts")
p.add_argument("--batch_size", type=int, default=5000)
p.add_argument("--min_free_mb", type=int, default=200)  # minimum free space to require before writing
p.add_argument("--resume", action="store_true", help="resume from existing progress (reads .progress)")
args = p.parse_args()

# -----------------------
# Helpers
# -----------------------
def norm_drug(s): return str(s).strip().lower().replace(" ", "_")
def norm_gene(s): return str(s).strip().upper().replace(" ", "_")
def norm_adr(s):  return str(s).strip().lower().replace(" ", "_")

def safe_shortest(u, v, cap=6):
    if (u not in G) or (v not in G):
        return cap
    try:
        return min(nx.shortest_path_length(G, u, v), cap)
    except nx.NetworkXNoPath:
        return cap

def cos_sim(u, v):
    nu = np.linalg.norm(u) + 1e-9
    nv = np.linalg.norm(v) + 1e-9
    return float(np.dot(u, v) / (nu * nv))

def ensure_free_space(path, min_free_mb):
    # check disk free bytes on the device containing path
    root = os.path.abspath(path)
    while not os.path.exists(root):
        root = os.path.dirname(root) or "/"
    du = shutil.disk_usage(root)
    free_mb = du.free // (1024 * 1024)
    return free_mb >= min_free_mb, free_mb

# -----------------------
# Load small metadata & graph
# -----------------------
print("📥 Loading inputs...")
triples = pd.read_csv(args.triples)
emb = np.load(args.emb)
nodes = pd.read_csv(args.nodes)
edges = pd.read_csv(args.kg_edges)

node_list = nodes["Node"].tolist()
node_to_idx = {n: i for i, n in enumerate(node_list)}

# knowledge graph
print("🗺️ Building KG graph...")
G = nx.Graph()
for _, r in edges.iterrows():
    s, t = r["src"], r["dst"]
    if s in node_to_idx and t in node_to_idx:
        G.add_edge(s, t)

# -----------------------
# PubMed (entity-focused low-memory) or skip
# -----------------------
if args.pubmed.lower() != "none":
    print("📚 Scanning PubMed for entities (low-memory mode)...")
    try:
        pub = pd.read_csv(args.pubmed)
    except Exception:
        print("⚠️ Could not load PubMed CSV; continuing without literature features.")
        pub = pd.DataFrame(columns=["Title", "Abstract", "PMID", "Year"])
    pub["text"] = (pub["Title"].fillna("") + " " + pub["Abstract"].fillna("")).str.lower()

    # unique terms only from triples (reduces memory)
    drug_terms = set(triples["drug"].astype(str).str.replace("_", " ").str.lower())
    gene_terms = set(triples["gene"].astype(str).str.replace("_", " ").str.lower())
    adr_terms  = set(triples["adr"].astype(str).str.replace("_", " ").str.lower())
    all_terms = drug_terms | gene_terms | adr_terms

    mention_cache = defaultdict(set)
    for _, row in tqdm(pub.iterrows(), total=len(pub), desc="Scanning PubMed"):
        text = row["text"]
        pmid = row.get("PMID", None)
        year = row.get("Year", 0)
        # check term containment (cheap since terms list is limited)
        for term in all_terms:
            if term and term in text:
                mention_cache[term].add((pmid, year))

    def lit_counts(drug, gene, adr):
        d = drug.replace("_", " ")
        g = gene.replace("_", " ")
        a = adr.replace("_", " ")
        md = mention_cache.get(d, set())
        mg = mention_cache.get(g, set())
        ma = mention_cache.get(a, set())
        if not md or not ma:
            return [0, 0, 0, 0, 0, 0]
        DA = len(md & ma)
        DG = len(md & mg) if g else 0
        GA = len(mg & ma) if g else 0
        DGA = len(md & mg & ma) if g else 0
        pmid_hits = DGA
        years = [y for _, y in md if isinstance(y, (int, float))]
        recent_year = max(years) if years else 0
        return [DA, DG, GA, DGA, pmid_hits, recent_year]
else:
    print("⚠️ Skipping PubMed features (use --pubmed none to explicitly skip).")
    def lit_counts(d,g,a): return [0,0,0,0,0,0]

# -----------------------
# Embedding accessor
# -----------------------
def get_vec(key):
    idx = node_to_idx.get(key)
    if idx is None:
        return np.zeros(emb.shape[1], dtype=np.float32)
    return emb[idx]

def prefixed(drug, gene, adr):
    return f"DRUG_{drug}", (f"GENE_{gene}" if gene else None), f"ADR_{adr}"

# -----------------------
# Prepare output directory and progress
# -----------------------
os.makedirs(args.out_dir, exist_ok=True)
progress_path = os.path.join(args.out_dir, ".progress.json")

if args.resume and os.path.exists(progress_path):
    with open(progress_path, "r") as fh:
        progress = json.load(fh)
    start_idx = int(progress.get("written", 0))
    part_idx = int(progress.get("part_idx", 0))
    print(f"🔁 Resuming from {start_idx} rows written, next part {part_idx}")
else:
    start_idx = 0
    part_idx = 0
    progress = {"written": 0, "part_idx": 0}

batch_size = int(args.batch_size)
buffer = []
written = int(progress.get("written", 0))
total = len(triples)
print(f"⚙️ Starting at row {start_idx} / {total}. Batch size = {batch_size}")

# -----------------------
# Streaming assembly loop
# -----------------------
for i, row in tqdm(enumerate(triples.itertuples(index=False)), total=total, initial=0, desc="Assembling"):
    if i < start_idx:
        continue

    d = norm_drug(row.drug)
    g = norm_gene(row.gene)
    a = norm_adr(row.adr)
    y = int(row.label) if hasattr(row, "label") else 1

    D, GENE, A = prefixed(d, g, a)
    vd = get_vec(D)
    vg = get_vec(GENE) if GENE else np.zeros_like(vd)
    va = get_vec(A)

    had_dg, had_da, had_ga = vd * vg, vd * va, vg * va
    l1_dg, l1_da, l1_ga = np.abs(vd - vg), np.abs(vd - va), np.abs(vg - va)

    cos_dg = cos_sim(vd, vg) if vg.any() else 0.0
    cos_da = cos_sim(vd, va)
    cos_ga = cos_sim(vg, va) if vg.any() else 0.0

    sp_dg = safe_shortest(D, GENE) if GENE else 6
    sp_da = safe_shortest(D, A)
    sp_ga = safe_shortest(GENE, A) if GENE else 6

    DA, DG, GA, DGA, pmids, recent_year = lit_counts(d, g, a)

    feats = np.concatenate([
        vd, vg, va,
        had_dg, had_da, had_ga,
        l1_dg, l1_da, l1_ga
    ]).astype(np.float32)

    rec = {
        "drug": d, "gene": g, "adr": a, "label": y,
        "cos_dg": cos_dg, "cos_da": cos_da, "cos_ga": cos_ga,
        "sp_dg": sp_dg, "sp_da": sp_da, "sp_ga": sp_ga,
        "lit_da": DA, "lit_dg": DG, "lit_ga": GA, "lit_dga": DGA,
        "lit_pmids": pmids, "lit_recent_year": recent_year
    }
    for j, val in enumerate(feats):
        rec[f"f{j}"] = float(val)

    buffer.append(rec)

    # flush
    if len(buffer) >= batch_size or i == total - 1:
        # check free space
        ok, free_mb = ensure_free_space(args.out_dir, args.min_free_mb)
        if not ok:
            print(f"\n⛔ Insufficient disk space: {free_mb} MB free (< {args.min_free_mb} MB). Aborting write.")
            print("You can free space or change --out_dir to a different disk with more space,")
            print("then re-run with --resume to continue.")
            # don't lose buffer: write it to a temporary local file? we abort to be safe.
            raise SystemExit(1)

        df_chunk = pd.DataFrame(buffer)
        df_chunk = df_chunk.replace([np.inf, -np.inf], 0).fillna(0)

        part_fname = os.path.join(args.out_dir, f"features_part_{part_idx:04d}.csv.gz")
        print(f"\n💾 Writing part {part_idx} ({len(df_chunk)} rows) -> {part_fname} ...")
        df_chunk.to_csv(part_fname, index=False, compression="gzip")
        written += len(buffer)
        buffer = []
        part_idx += 1

        # update progress
        progress = {"written": written, "part_idx": part_idx}
        with open(progress_path, "w") as fh:
            json.dump(progress, fh)

        # quick status
        print(f"✅ Total written so far: {written:,} / {total}")

print(f"\n🎉 Finished processing {written:,} triples. Parts available in: {args.out_dir}")
print("Combine parts when ready (see notes).")
