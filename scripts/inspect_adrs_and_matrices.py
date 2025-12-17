# scripts/inspect_adrs_and_matrices.py
import pandas as pd, os

files = [
    ("filtered_adrs", "data/filtered_adrs.csv"),
    ("adr_mesh_similarity", "data/adr_mesh_similarity.csv"),
    ("adr_gda_similarity", "data/adr_gda_similarity.csv")
]

for name, path in files:
    print(f"\n--- {name}: {path} ---")
    if not os.path.exists(path):
        print("MISSING")
        continue
    try:
        df = pd.read_csv(path, nrows=10)
        print("Preview (first 10 rows):")
        print(df.head(10))
        # try reading full table for matrices (may be big) but only list indexes/cols for csvs
        if "mesh" in name or "gda" in name:
            full = pd.read_csv(path, index_col=0)
            print("Shape:", full.shape)
            print("Index sample:", list(full.index[:10]))
            print("Columns sample:", list(full.columns[:10]))
    except Exception as e:
        print("Error reading:", e)
