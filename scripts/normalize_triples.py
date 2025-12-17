import pandas as pd
import re
from tqdm import tqdm

print("🔹 Loading extracted triples...")
triples = pd.read_csv("output/triples.csv")
print(f"✅ Loaded {len(triples)} triples")

# --- Load mapping resources ---
print("🔹 Loading mapping files...")

sider = pd.read_csv("data_raw/SIDER/drug_names.tsv", sep="\t", header=None, names=["STITCH_ID", "DrugName"])
sider["DrugName"] = sider["DrugName"].str.lower().str.strip()
sider["STITCH_ID"] = sider["STITCH_ID"].str.replace("CID", "", regex=False)

adrs = pd.read_csv("data/filtered_adrs.csv")
adrs["MeSH_Name"] = adrs["MeSH_Name"].str.lower().str.strip()

# Optional: load HGNC gene symbols if available
try:
    genes = pd.read_csv("data_raw/HGNC/hgnc_complete_set.txt", sep="\t", usecols=["symbol"])
    gene_symbols = set(genes["symbol"].str.lower())
    print(f"✅ Loaded {len(gene_symbols)} HGNC genes")
except Exception:
    gene_symbols = set()
    print("⚠️ HGNC file not found — using gene symbol text matching only")

# --- Normalize text ---
def clean_text(x):
    if pd.isna(x): return ""
    return re.sub(r"[^a-z0-9\- ]", "", str(x).lower().strip())

triples["Drug"] = triples["Drug"].apply(clean_text)
triples["Gene"] = triples["Gene"].apply(clean_text)
triples["ADR"] = triples["ADR"].apply(clean_text)

# --- Map drugs ---
drug_map = dict(zip(sider["DrugName"], sider["STITCH_ID"]))
triples["DrugID"] = triples["Drug"].map(drug_map)

# --- Map ADRs ---
adr_map = dict(zip(adrs["MeSH_Name"], adrs["MeSH_ID"]))
triples["MeSH_ID"] = triples["ADR"].map(adr_map)

# --- Validate genes ---
triples["GeneValid"] = triples["Gene"].apply(lambda x: x in gene_symbols if gene_symbols else True)

# --- Filter ---
valid_triples = triples[
    triples["DrugID"].notna() &
    triples["MeSH_ID"].notna() &
    triples["GeneValid"]
].drop_duplicates()

print(f"✅ Normalized {len(valid_triples)} valid triples (Drug–Gene–ADR)")

# --- Save ---
valid_triples.to_csv("data/triples_normalized.csv", index=False)
print("💾 Saved → data/triples_normalized.csv")
