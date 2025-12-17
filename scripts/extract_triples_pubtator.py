import requests
import pandas as pd
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm

INPUT_FILE = "data/pubmed_results_large.csv"
OUTPUT_FILE = "output/triples.csv"
SAVE_EVERY = 25
BATCH_SIZE = 100

PUBTATOR_TEXT_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids="

print("🔹 Loading PubMed results...")
df = pd.read_csv(INPUT_FILE)
pmids = df["PMID"].astype(str).unique().tolist()
print(f"✅ Loaded {len(pmids)} PubMed abstracts")

all_triples = []
seen = set()

def fetch_pubtator_xml(batch_pmids):
    ids = ",".join(batch_pmids)
    url = PUBTATOR_TEXT_URL + ids
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f"⚠️ Error {r.status_code} fetching {len(batch_pmids)} PMIDs")
            return []
        return r.text
    except Exception as e:
        print(f"⚠️ Fetch error: {e}")
        return None

def parse_pubtator_xml(xml_text):
    triples = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for doc in root.findall(".//document"):
        pmid_elem = doc.find("id")
        if pmid_elem is None:
            continue
        pmid = pmid_elem.text

        chemicals, genes, adrs = set(), set(), set()

        for passage in doc.findall(".//passage"):
            for annotation in passage.findall("annotation"):
                type_elem = annotation.find("infon[@key='type']")
                text_elem = annotation.find("text")
                if type_elem is None or text_elem is None:
                    continue
                t = type_elem.text.lower()
                text = text_elem.text.lower()
                if t in ["chemical", "drug", "compound"]:
                    chemicals.add(text)
                elif t == "gene":
                    genes.add(text)
                elif t in ["disease", "side effect", "phenotype"]:
                    adrs.add(text)

        for d in chemicals:
            for g in genes:
                for a in adrs:
                    triples.append((pmid, d, g, a))

    return triples

# --- Main loop ---
for i in tqdm(range(0, len(pmids), BATCH_SIZE), desc="Extracting PubTator triples"):
    batch = pmids[i:i+BATCH_SIZE]
    xml_data = fetch_pubtator_xml(batch)
    if not xml_data:
        continue
    batch_triples = parse_pubtator_xml(xml_data)
    new_triples = [t for t in batch_triples if t not in seen]
    all_triples.extend(new_triples)
    seen.update(new_triples)

    if (i // BATCH_SIZE + 1) % SAVE_EVERY == 0:
        pd.DataFrame(all_triples, columns=["PMID", "Drug", "Gene", "ADR"]).to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Saved progress ({len(all_triples)} triples so far)")
    time.sleep(1)

# --- Final save ---
pd.DataFrame(all_triples, columns=["PMID", "Drug", "Gene", "ADR"]).to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Extracted {len(all_triples)} triples total → {OUTPUT_FILE}")
