import pandas as pd
import requests
import time
import os
import json

API_KEY = "451a38b9-ab75-497c-9845-9c676c8c7056"
SIDER_FILE = "data_raw/SIDER/meddra_all_se.tsv.gz"
OUT_FILE = "data/filtered_adrs.csv"
CACHE_FILE = "data/umls_mesh_cache.json"

# Load cache if exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

# Load SIDER
df = pd.read_csv(SIDER_FILE, sep="\t", header=None, low_memory=False)
df.columns = [
    "STITCH_ID_flat",
    "STITCH_ID_stereo",
    "UMLS_ID",
    "MedDRA_type",
    "SideEffectName",
    "Frequency"
]
print("Loaded", len(df), "side effect entries")

# Keep unique CUIs and names
adrs = df.groupby("UMLS_ID").first().reset_index()

print("Unique ADR CUIs:", len(adrs))  # should be ~7943

# Function to query BioPortal
def umls_to_mesh(cui):
    if cui in cache:
        return cache[cui]

    url = f"http://data.bioontology.org/search?q={cui}&require_exact_match=true&ontologies=MESH&apikey={API_KEY}"

    for attempt in range(5):  # retry up to 5 times
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("collection"):
                    item = data["collection"][0]
                    mesh_id = item.get("@id").split("/")[-1]
                    label = item.get("prefLabel")
                    cache[cui] = (mesh_id, label)
                    return mesh_id, label
                else:
                    cache[cui] = (None, None)
                    return None, None
            else:
                print(f"Error {r.status_code} for {cui}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Connection error for {cui}: {e}, retrying...")

        time.sleep(2 ** attempt)  # exponential backoff

    cache[cui] = (None, None)
    return None, None

# Map CUIs
mesh_ids, mesh_names = [], []
for i, row in adrs.iterrows():
    cui = row["UMLS_ID"]
    mesh_id, mesh_name = umls_to_mesh(cui)
    mesh_ids.append(mesh_id)
    mesh_names.append(mesh_name if mesh_name else row["SideEffectName"])

    # Progress + save cache every 50 entries
    if i % 50 == 0:
        print(f"Processed {i}/{len(adrs)}")
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)

# Save results
adrs["MeSH_ID"] = mesh_ids
adrs["MeSH_Name"] = mesh_names
adrs = adrs.dropna(subset=["MeSH_ID"]).drop_duplicates()
adrs.to_csv(OUT_FILE, index=False)

with open(CACHE_FILE, "w") as f:
    json.dump(cache, f)

print(f"Saved {OUT_FILE} with {len(adrs)} mapped rows")
