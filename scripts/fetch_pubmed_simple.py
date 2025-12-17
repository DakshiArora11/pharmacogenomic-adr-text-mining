from Bio import Entrez
import csv, time, argparse

def fetch(query, retmax=200, email='you@example.com', out='data/pubmed_results.csv'):
    Entrez.email = email
    print("Searching PubMed for:", query)
    handle = Entrez.esearch(db='pubmed', term=query, retmax=retmax)
    record = Entrez.read(handle)
    ids = record['IdList']
    print(f'Found {len(ids)} PMIDs, fetching...')
    rows=[]
    for i in range(0,len(ids),200):
        batch = ids[i:i+200]
        h = Entrez.efetch(db='pubmed', id=','.join(batch), rettype='xml', retmode='xml')
        recs = Entrez.read(h)
        for art in recs.get('PubmedArticle', []):
            pmid = art['MedlineCitation']['PMID']
            title = art['MedlineCitation']['Article'].get('ArticleTitle', '')
            abstract = ''
            if 'Abstract' in art['MedlineCitation']['Article']:
                A = art['MedlineCitation']['Article']['Abstract'].get('AbstractText', '')
                if isinstance(A, list):
                    abstract = ' '.join(A)
                else:
                    abstract = str(A)
            rows.append({'pmid': str(pmid), 'title': title, 'abstract': abstract})
        time.sleep(0.35)  # be gentle with NCBI
    with open(out, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['pmid','title','abstract'])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--query', required=True)
    p.add_argument('--retmax', type=int, default=200)
    p.add_argument('--email', default='you@example.com')
    p.add_argument('--out', default='data/pubmed_results.csv')
    args=p.parse_args()
    fetch(args.query, args.retmax, args.email, args.out)
