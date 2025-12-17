import csv, argparse
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def read_pubmed(csvfile):
    rows=[]
    with open(csvfile, encoding='utf8') as f:
        reader=csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def extract(rows, drug_terms, gene_terms, adr_terms, out='output/triples.csv'):
    triples=[]
    for r in rows:
        text = (r.get('title') or '') + '. ' + (r.get('abstract') or '')
        for s in sent_tokenize(text):
            s_low = s.lower()
            found_drugs = [d for d in drug_terms if d.lower() in s_low]
            found_genes = [g for g in gene_terms if g.lower() in s_low]
            found_adrs = [a for a in adr_terms if a.lower() in s_low]
            if found_drugs and (found_genes or found_adrs):
                triples.append({
                    'pmid': r['pmid'], 'sentence': s.strip(),
                    'drugs': '|'.join(found_drugs),
                    'genes': '|'.join(found_genes),
                    'adrs': '|'.join(found_adrs)
                })
    with open(out, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['pmid','sentence','drugs','genes','adrs'])
        writer.writeheader()
        writer.writerows(triples)
    print("Wrote", len(triples), "candidate sentences to", out)
    return triples

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pubmed_csv', default='data/pubmed_results.csv')
    p.add_argument('--out', default='output/triples.csv')
    args=p.parse_args()
    drug_terms=['warfarin','coumadin']
    gene_terms=['cyp2c9','vkorc1']
    adr_terms=['bleeding','haemorrhage','hemorrhage','bleed','adverse','side effect','toxicity']
    rows = read_pubmed(args.pubmed_csv)
    extract(rows, drug_terms, gene_terms, adr_terms, args.out)
