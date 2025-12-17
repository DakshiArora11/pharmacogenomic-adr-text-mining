import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
import requests
from tqdm import tqdm

# --- Load filtered drug list ---
filtered = pd.read_csv("../data/filtered_drugs_lincs.csv")
print(f"Loaded {len(filtered)} drugs")

# Drop duplicates
filtered = filtered.drop_duplicates(subset=['pubchem_cid'])
filtered = filtered[filtered['pubchem_cid'].notna()]

# Get PubChem SMILES for each CID
smiles_dict = {}
for cid in tqdm(filtered['pubchem_cid'], desc="Fetching SMILES"):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            smi = r.text.strip()
            smiles_dict[cid] = smi
        else:
            smiles_dict[cid] = None
    except Exception:
        smiles_dict[cid] = None

filtered['smiles'] = filtered['pubchem_cid'].map(smiles_dict)

# Compute fingerprints
fps = {}
for cid, smi in tqdm(smiles_dict.items(), desc="Computing fingerprints"):
    if smi:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)  # MACCS fingerprint (166 bits)
            fps[cid] = fp
        else:
            fps[cid] = None
    else:
        fps[cid] = None

# Build similarity matrix
cids = list(filtered['pubchem_cid'])
n = len(cids)
sim_matrix = pd.DataFrame(0.0, index=cids, columns=cids)

for i, cid1 in enumerate(tqdm(cids, desc="Computing similarity")):
    fp1 = fps[cid1]
    if fp1 is None:
        continue
    for j, cid2 in enumerate(cids[i:], start=i):
        fp2 = fps[cid2]
        if fp2 is None:
            continue
        sim = DataStructs.FingerprintSimilarity(fp1, fp2)
        sim_matrix.iloc[i, j] = sim
        sim_matrix.iloc[j, i] = sim  # symmetric

# Save similarity matrix
sim_matrix.to_csv("../data/drug_structure_similarity.csv")
print("Saved ../data/drug_structure_similarity.csv with shape", sim_matrix.shape)