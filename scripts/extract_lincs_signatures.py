# extract_lincs_signatures.py
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

# --- Load filtered drug list ---
filtered = pd.read_csv("../data/filtered_drugs_lincs.csv")
pert_ids = list(filtered['pert_id'].unique())
print(f"Extracting signatures for {len(pert_ids)} pert_ids")

# --- Path to the GCTX file ---
gctx_file = "../data_raw/LINCS/GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx"

# --- Load the GCTX file ---
gctoo = parse(gctx_file)
df = gctoo.data_df  # rows = genes, columns = distil_id
print("Full matrix shape:", df.shape)

# Transpose so rows = distil_id, columns = 978 genes
df_T = df.T

# --- Load signature info to map distil_id -> pert_id ---
sig_info = pd.read_csv(
    "../data_raw/LINCS/GSE92742_Broad_LINCS_sig_info.txt.gz",
    sep="\t",
    low_memory=False
)

# Keep only rows with pert_ids of interest
sig_info = sig_info[sig_info['pert_id'].isin(pert_ids)]

# Make a mapping distil_id → pert_id
# (distil_id column has the actual column names of the GCTX file)
sig_map = sig_info.set_index('distil_id')['pert_id'].to_dict()

# Filter the transposed dataframe to only those distil_ids we have
common_distil_ids = [distil_id for distil_id in df_T.index if distil_id in sig_map]
print("Common distil_id count:", len(common_distil_ids))

df_filtered = df_T.loc[common_distil_ids]

# Add pert_id column
df_filtered['pert_id'] = df_filtered.index.map(sig_map)

# Average replicates per pert_id to get one signature per drug
df_mean = df_filtered.groupby('pert_id').mean()

# Save to file
df_mean.to_csv("../data/lincs_signatures.csv")
print("Saved ../data/lincs_signatures.csv with shape", df_mean.shape)
