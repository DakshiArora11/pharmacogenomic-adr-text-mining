# build_drug_cgi_similarity.py
# Compute Drug–Gene Interaction (CGI) similarity using CTD

import pandas as pd
from tqdm import tqdm

# --- Load filtered drugs list ---
filtered = pd.read_csv("../data/filtered_drugs_lincs.csv")
print(f"Loaded {len(filtered)} drugs from filtered list")

# figure out which column has drug names
if "drug_name" in filtered.columns:
    drug_names = set(filtered["drug_name"].str.lower())
elif "pert_iname" in filtered.columns:
    drug_names = set(filtered["pert_iname"].str.lower())
else:
    # fall back to any column with names
    drug_names = set(filtered.iloc[:,0].astype(str).str.lower())

# --- Load CTD chemical-gene interactions ---
path = "../data_raw/CTD/CTD_chem_gene_ixns.tsv.gz"

colnames = [
    "ChemicalName",
    "ChemicalID",
    "CasRN",
    "GeneSymbol",
    "GeneID",
    "GeneForms",
    "Organism",
    "OrganismID",
    "Interaction",
    "InteractionActions",
    "PubMedIDs"
]

ctd_cgi = pd.read_csv(
    path,
    sep="\t",
    comment="#",
    header=None,
    names=colnames,
    low_memory=False
)

print("CTD loaded with shape:", ctd_cgi.shape)

# filter CTD rows where ChemicalName matches our filtered drug names
ctd_cgi["ChemicalName_lower"] = ctd_cgi["ChemicalName"].str.lower()
ctd_filtered = ctd_cgi[ctd_cgi["ChemicalName_lower"].isin(drug_names)]
print("Filtered CTD shape:", ctd_filtered.shape)

# build mapping: ChemicalName -> set of genes
chem_to_genes = {}
for chem, group in ctd_filtered.groupby("ChemicalName_lower"):
    chem_to_genes[chem] = set(group["GeneSymbol"].dropna().astype(str))

chem_ids = list(chem_to_genes.keys())
n = len(chem_ids)
print(f"Computing similarity for {n} chemicals")

# compute Jaccard similarity
sim_matrix = pd.DataFrame(0.0, index=chem_ids, columns=chem_ids)

for i in tqdm(range(n)):
    for j in range(i, n):
        g1 = chem_to_genes[chem_ids[i]]
        g2 = chem_to_genes[chem_ids[j]]
        if not g1 or not g2:
            sim = 0.0
        else:
            sim = len(g1 & g2) / len(g1 | g2)
        sim_matrix.iat[i, j] = sim
        sim_matrix.iat[j, i] = sim

# save
sim_matrix.to_csv("../data/drug_cgi_similarity.csv")
print(f"Saved ../data/drug_cgi_similarity.csv with shape {sim_matrix.shape}")
