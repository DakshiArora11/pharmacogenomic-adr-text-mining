# scripts/fetch_pubmed_bulk.py
import time
import csv
from tqdm import tqdm
from Bio import Entrez

# --------------------------------------------
# ✅ CONFIGURATION
# --------------------------------------------

# Example focused drugs for ADR extraction
DRUG_KEYWORDS = [
    "warfarin", "ibuprofen", "cisplatin", "metformin",
    "clopidogrel", "tamoxifen", "carbamazepine", "fluoxetine"
]

# How many results per drug to fetch (adjust as needed)
MAX_RESULTS_PER_DRUG = 3000

OUTPUT_FILE = "data/pubmed_results_large.csv"


# --------------------------------------------
# 🔍 Function to fetch PubMed IDs
# --------------------------------------------
def fetch_pubmed_ids(query, max_results=1000):
    """Return PubMed IDs for a search query"""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        retmode="xml"
    )
    results = Entrez.read(handle)
    handle.close()
    return results["IdList"]


# --------------------------------------------
# 📄 Function to fetch paper details
# --------------------------------------------
def fetch_details(id_list):
    """Fetch title and abstract for a list of PubMed IDs"""
    ids = ",".join(id_list)
    handle = Entrez.efetch(
        db="pubmed",
        id=ids,
        rettype="abstract",
        retmode="xml"
    )
    records = Entrez.read(handle)
    handle.close()
    return records


# --------------------------------------------
# 🚀 Main fetch loop
# --------------------------------------------
def main():
    print(f"🔹 Starting PubMed retrieval for {len(DRUG_KEYWORDS)} drugs...")
    all_rows = []
    seen_pmids = set()

    for drug in DRUG_KEYWORDS:
        query = f"{drug} AND (gene OR polymorphism OR side effect OR adverse reaction)"
        print(f"\n🔸 Querying PubMed for: {query}")

        try:
            pmids = fetch_pubmed_ids(query, MAX_RESULTS_PER_DRUG)
        except Exception as e:
            print(f"⚠️ Error fetching IDs for {drug}: {e}")
            continue

        print(f"   Found {len(pmids)} papers for {drug}")

        # Fetch in batches of 200
        for i in tqdm(range(0, len(pmids), 200), desc=f"Fetching {drug} abstracts"):
            batch = pmids[i:i+200]
            try:
                records = fetch_details(batch)
                for rec in records["PubmedArticle"]:
                    pmid = rec["MedlineCitation"]["PMID"]
                    title = rec["MedlineCitation"]["Article"].get("ArticleTitle", "")
                    abstract = ""
                    if "Abstract" in rec["MedlineCitation"]["Article"]:
                        abstract = " ".join(rec["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])

                    if pmid not in seen_pmids and abstract.strip():
                        all_rows.append([pmid, drug, title, abstract])
                        seen_pmids.add(pmid)

                time.sleep(0.3)  # be kind to NCBI API

            except Exception as e:
                print(f"⚠️ Error fetching batch: {e}")
                time.sleep(3)

    # Save results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["PMID", "Drug", "Title", "Abstract"])
        writer.writerows(all_rows)

    print(f"\n✅ Saved {len(all_rows)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
