import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# -----------------------------
# Load filtered drugs with LINCS info
# -----------------------------
filtered = pd.read_csv("../data/filtered_drugs_lincs.csv")

# -----------------------------
# 1. Chemical Fingerprints
# -----------------------------
# For each drug_name, fetch SMILES from PubChem
# We'll use requests to query PubChem REST API
import requests

def get_smiles_pubchem(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=10)
        j = r.json()
        return j['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception:
        return None

filtered['smiles'] = filtered['drug_name'].apply(get_smiles_pubchem)
filtered.to_csv("../data/filtered_drugs_with_smiles.csv", index=False)

# Compute fingerprints
fps = {}
for idx, row in filtered.iterrows():
    if row['smiles']:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps[row['drug_name']] = fp

# Pairwise Tanimoto
drug_names = list(fps.keys())
n = len(drug_names)
chem_sim = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        chem_sim[i,j] = DataStructs.TanimotoSimilarity(fps[drug_names[i]], fps[drug_names[j]])

chem_sim_df = pd.DataFrame(chem_sim, index=drug_names, columns=drug_names)
chem_sim_df.to_csv("../data/drug_chem_similarity.csv")
print("Saved ../data/drug_chem_similarity.csv")

# -----------------------------
# 2. Gene Expression Similarity
# -----------------------------
expr = pd.read_csv("../data/lincs_signatures.csv", index_col=0)
expr_names = expr.index  # pert_id
# We need mapping pert_id → drug_name
pert_map = filtered.set_index('pert_id')['drug_name'].to_dict()

expr.index = expr.index.map(pert_map)
expr = expr.groupby(expr.index).mean()

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
ge_sim = cosine_similarity(expr)
ge_sim_df = pd.DataFrame(ge_sim, index=expr.index, columns=expr.index)
ge_sim_df.to_csv("../data/drug_ge_similarity.csv")
print("Saved ../data/drug_ge_similarity.csv")

# -----------------------------
# 3. Chemical–Gene Interaction (CGI) Similarity
# -----------------------------
ctd = pd.read_csv("../data_raw/CTD/CTD_chem_gene_ixns.tsv.gz", sep="\t", comment="#", compression='gzip')
ctd = ctd[['ChemicalName','GeneSymbol']].dropna().drop_duplicates()

# Build a dict: drug_name → set of genes
gene_sets = {}
for drug in filtered['drug_name']:
    genes = set(ctd[ctd['ChemicalName'].str.lower()==drug.lower()]['GeneSymbol'])
    if genes:
        gene_sets[drug] = genes

drugs = list(gene_sets.keys())
m = len(drugs)
cgi_sim = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        gi = gene_sets[drugs[i]]
        gj = gene_sets[drugs[j]]
        inter = len(gi & gj)
        union = len(gi | gj)
        cgi_sim[i,j] = inter/union if union>0 else 0

cgi_sim_df = pd.DataFrame(cgi_sim, index=drugs, columns=drugs)
cgi_sim_df.to_csv("../data/drug_cgi_similarity.csv")
print("Saved ../data/drug_cgi_similarity.csv")
