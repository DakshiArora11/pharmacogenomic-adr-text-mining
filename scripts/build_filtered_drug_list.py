import pandas as pd

# -------------------------------
# Load SIDER drug names
# -------------------------------
# This file maps STITCH IDs to drug names.
# In SIDER v4.1 it’s named "drug_names.tsv" and looks like:
# STITCH_ID   DrugName
drug_names = pd.read_csv("../data_raw/SIDER/drug_names.tsv", sep="\t", header=None, names=['stitch_id','drug_name'])
drug_names['drug_name_lower'] = drug_names['drug_name'].str.lower()

print(f"Loaded {len(drug_names)} SIDER drug-name mappings")

# -------------------------------
# Load LINCS perturbation info
# -------------------------------
pert = pd.read_csv("../data_raw/LINCS/GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t")
pert['pert_iname_lower'] = pert['pert_iname'].str.lower()

print(f"Loaded {len(pert)} LINCS perturbations")

# -------------------------------
# Intersect by drug name (lowercase)
# -------------------------------
common_names = set(drug_names['drug_name_lower']).intersection(set(pert['pert_iname_lower']))
print(f"Found {len(common_names)} overlapping drug names between SIDER and LINCS")

# Filter both tables to common names
sider_common = drug_names[drug_names['drug_name_lower'].isin(common_names)]
pert_common = pert[pert['pert_iname_lower'].isin(common_names)]

# Merge to get STITCH ID + LINCS pert_id together
merged = sider_common.merge(
    pert_common,
    left_on='drug_name_lower',
    right_on='pert_iname_lower',
    suffixes=('_sider','_lincs')
)

# Keep only useful columns
merged = merged[['stitch_id','drug_name','pert_id','pert_iname','pubchem_cid']]
merged.to_csv("../data/filtered_drugs_lincs.csv", index=False)

print(f"Saved merged list to ../data/filtered_drugs_lincs.csv with {len(merged)} rows")
