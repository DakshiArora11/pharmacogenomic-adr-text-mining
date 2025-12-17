import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument('--triples_csv', default='output/triples.csv')
p.add_argument('--emb_out', default='output/node_embeddings.csv')
args = p.parse_args()

print("Reading triples from", args.triples_csv)
df = pd.read_csv(args.triples_csv)

# Build graph
G = nx.Graph()

for _, row in df.iterrows():
    drugs = [d.strip() for d in str(row['drugs']).split('|') if d and d != 'nan']
    genes = [g.strip() for g in str(row['genes']).split('|') if g and g != 'nan']
    adrs = [a.strip() for a in str(row['adrs']).split('|') if a and a != 'nan']

    # create nodes with type prefixes
    for d in drugs:
        G.add_node(f"D:{d}", type='Drug')
    for g in genes:
        G.add_node(f"G:{g}", type='Gene')
    for a in adrs:
        G.add_node(f"A:{a}", type='ADR')

    # edges drug-gene
    for d in drugs:
        for g in genes:
            G.add_edge(f"D:{d}", f"G:{g}", relation='mentions', pmid=row['pmid'])
    # edges drug-ADR
    for d in drugs:
        for a in adrs:
            G.add_edge(f"D:{d}", f"A:{a}", relation='mentions', pmid=row['pmid'])
    # optionally gene-ADR edges
    for g in genes:
        for a in adrs:
            G.add_edge(f"G:{g}", f"A:{a}", relation='mentions', pmid=row['pmid'])

print("Graph has", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges")

# Compute node2vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

# Save embeddings
rows=[]
for node in G.nodes():
    vec = model.wv[node]
    rows.append([node] + vec.tolist())
emb_df = pd.DataFrame(rows)
emb_df.to_csv(args.emb_out, index=False, header=False)
print("Saved embeddings to", args.emb_out)
