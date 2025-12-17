import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

# Load your filtered drug list
filtered = pd.read_csv("../data/filtered_drugs_lincs.csv")
pert_ids = list(filtered['pert_id'].unique())
print("Filtered pert_ids:", len(pert_ids))

# Load sig_info
sig_info = pd.read_csv(
    "../data_raw/LINCS/GSE92742_Broad_LINCS_sig_info.txt.gz",
    sep="\t",
    low_memory=False
)
print("Total sig_info rows:", sig_info.shape)

# Filter sig_info
sig_info = sig_info[sig_info['pert_id'].isin(pert_ids)]
print("sig_info rows after filter:", sig_info.shape)

# Show a few pert_id and sig_id to inspect
print(sig_info[['pert_id','sig_id']].head())

# Load GCTX meta
gctx_file = "../data_raw/LINCS/GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx"
gctoo = parse(gctx_file)
df_T = gctoo.data_df.T
print("GCTX columns count (sig_id):", len(df_T))

# Now test intersection
sig_ids = sig_info['sig_id'].tolist()
common = df_T.index.intersection(sig_ids)
print("Common sig_id count:", len(common))
