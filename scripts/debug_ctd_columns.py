import gzip

ctd_path = "../data_raw/CTD/CTD_genes_diseases.tsv.gz"

# --- Auto-detect header line ---
header_line = None
with gzip.open(ctd_path, "rt", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if line.startswith("GeneSymbol"):
            header_line = i
            print("Header found at line index:", i)
            break

if header_line is None:
    raise RuntimeError("Could not find CTD header row in file")
    