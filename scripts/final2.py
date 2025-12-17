import os
import json
import re
import difflib
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="DGANet-Rich Clinical Decision Support System", page_icon="🧠", layout="wide")

# Paths
FEATURES_PATH = os.getenv("DGANET_FEATURES_PATH", r"D:/dganet_features/features_rich_full.csv.gz")
PRED_PATH = os.getenv("DGANET_PREDICTIONS_PATH", r"D:/dganet_features/predictions_rich.csv")
PUBMED_PATH = "data/pubmed_results_large.csv"
MESH_CACHE = "data/umls_mesh_cache.json"
HISTORY_FILE = "search_history.json"

TOPK_MAIN = 20
GLOBAL_MAX = 60

# Drug synonyms for common misspellings and brand names
DRUG_SYNONYMS = {
    "dispin": "aspirin",
    "disprin": "aspirin",
    "asa": "aspirin",
    "ecotrin": "aspirin",
    "bufferin": "aspirin",
    "noantipian": "naratriptan",
    "workfarin": "warfarin",
    "workfarin": "warfarin",
    "autoregulation": "naratriptan",  # Based on your recent searches
    "caption": "naratriptan",  # Based on your recent searches
    "examination": "naratriptan",  # Based on your recent searches
}

# ---------------------- THEME / CSS ----------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#f8fbff,#e8f0fe); }
h1 {text-align:center;font-weight:700;background:-webkit-linear-gradient(45deg,#1565c0,#42a5f5);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5em;}
.section-title {font-size:1.15rem;font-weight:700;margin:0.5rem 0;color:#0d47a1;}
.query-card {background:#fff;padding:0.8rem 1rem;border-left:5px solid #2196f3;border-radius:10px;
             box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:1rem;}
.footer-note{text-align:center;color:#666;margin-top:1.5rem;font-size:0.85rem;}
.metric-card {background:#fff;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.08);padding:18px;text-align:center;}
.metric-title {color:#607d8b;font-size:0.85rem;margin-bottom:6px;}
.metric-big {font-size:1.75rem;font-weight:800;color:#0d47a1;}
.metric-sub {font-size:0.85rem;color:#607d8b;}
.smallnote {font-size:0.8rem;color:#607d8b;}
.team-credits {text-align:center;color:#444;margin-top:0.5rem;font-size:0.8rem;font-style:italic;}
</style>
""", unsafe_allow_html=True)

# ---------------------- UTILS ----------------------
def load_json_safe(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def normalize(s): 
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

@st.cache_data(show_spinner=False)
def load_core():
    rich = pd.read_csv(FEATURES_PATH, compression="gzip")
    preds = pd.read_csv(PRED_PATH)
    
    if "prediction_score" not in preds.columns:
        st.error("❌ predictions_rich.csv must have column 'prediction_score'")
        st.stop()
    
    if len(preds) != len(rich):
        st.warning("⚠️ Length mismatch between features and predictions. Merging by row order.")
    
    rich = rich.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    rich["prediction_score"] = preds["prediction_score"]
    
    pubmed = pd.read_csv(PUBMED_PATH) if os.path.exists(PUBMED_PATH) else pd.DataFrame()
    mesh_map = load_json_safe(MESH_CACHE, {})
    
    return rich, pubmed, mesh_map

def fuzzy_match_drug(drug_text, drug_list):
    """Fuzzy match for drugs with synonym support"""
    drug_lower = drug_text.lower().strip()
    
    # First check synonyms
    if drug_lower in DRUG_SYNONYMS:
        canonical_name = DRUG_SYNONYMS[drug_lower]
        if canonical_name in drug_list:
            return canonical_name, 0.95  # High confidence for known synonyms
    
    # Then exact match
    dnorm = normalize(drug_text)
    for d in drug_list:
        if normalize(d) == dnorm:
            return d, 1.0
    
    # Then fuzzy match
    best = difflib.get_close_matches(drug_text, drug_list, n=1, cutoff=0.6)
    if best:
        score = difflib.SequenceMatcher(None, drug_text.lower(), best[0].lower()).ratio()
        return best[0], score
    
    return None, 0.0

def fuzzy_match_gene(gene_text, gene_list):
    """Fuzzy match for genes with better handling of various formats"""
    if not gene_text:
        return None, 0.0
    
    gene_text_upper = gene_text.upper().strip()
    
    # First try exact match (case insensitive)
    for gene in gene_list:
        if str(gene).upper() == gene_text_upper:
            return gene, 1.0
    
    # Handle gene name variations (with dashes, numbers, etc.)
    gene_clean = re.sub(r'[^a-zA-Z0-9]', '', gene_text_upper)
    for gene in gene_list:
        gene_clean_compare = re.sub(r'[^a-zA-Z0-9]', '', str(gene).upper())
        if gene_clean_compare == gene_clean:
            return gene, 0.95
    
    # Then try close matches
    best = difflib.get_close_matches(gene_text, gene_list, n=1, cutoff=0.7)
    if best:
        score = difflib.SequenceMatcher(None, gene_text.upper(), str(best[0]).upper()).ratio()
        return best[0], score
    
    return None, 0.0

def mesh_name(code, mesh_map, fetch=False):
    """
    Map ADR code/text to pretty name. 
    If fetch=True and looks like MeSH ID, try NLM API once and persist.
    """
    if not isinstance(code, str):
        return str(code)
    
    # Check if it's already a readable name (not a D-code)
    if not re.fullmatch(r"D\d{6}", code.strip()):
        # It's already a name, just clean it up
        return code.replace("_", " ").replace("-", " ").title()
    
    # Check cache first
    if code in mesh_map and mesh_map[code]:
        return mesh_map[code]
    
    # If not fetching, return the code as is
    if not fetch:
        return code
    
    # Try to fetch from NLM MeSH API
    try:
        # First try the lookup API to get the concept ID
        lookup_url = f"https://id.nlm.nih.gov/mesh/lookup/descriptor?label={code}&match=exact"
        response = requests.get(lookup_url, timeout=10)
        
        if response.status_code == 200 and response.json():
            # Get the resource URI from the lookup result
            lookup_data = response.json()
            if lookup_data and len(lookup_data) > 0:
                resource_uri = lookup_data[0].get('resource', '')
                
                if resource_uri:
                    # Extract the ID from the resource URI
                    mesh_id = resource_uri.split('/')[-1]
                    
                    # Now get the detailed information
                    detail_url = f"https://id.nlm.nih.gov/mesh/lookup/details?descriptor={mesh_id}"
                    detail_response = requests.get(detail_url, timeout=10)
                    
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        if detail_data and 'terms' in detail_data and detail_data['terms']:
                            # Get the preferred term
                            preferred_term = None
                            for term in detail_data['terms']:
                                if term.get('preferred', False):
                                    preferred_term = term.get('name', code)
                                    break
                            
                            # If no preferred term, use the first one
                            if not preferred_term and detail_data['terms']:
                                preferred_term = detail_data['terms'][0].get('name', code)
                            
                            if preferred_term:
                                mesh_map[code] = preferred_term
                                save_json(MESH_CACHE, mesh_map)
                                return preferred_term
        
        # Fallback: try the simpler label endpoint
        label_url = f"https://id.nlm.nih.gov/mesh/lookup/label?label={code}&match=exact"
        label_response = requests.get(label_url, timeout=10)
        
        if label_response.status_code == 200:
            label_data = label_response.json()
            if label_data and len(label_data) > 0 and 'label' in label_data[0]:
                label = label_data[0]['label']
                mesh_map[code] = label
                save_json(MESH_CACHE, mesh_map)
                return label
                
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not fetch MeSH name for {code}: {str(e)}")
    
    return code  # Return original code if all fails

def apply_mesh_names(series, mesh_map, fetch=False):
    """Apply MeSH name conversion to a pandas series with progress tracking"""
    if fetch:
        st.sidebar.info("🔄 Fetching MeSH names from NLM...")
    
    result = series.apply(lambda x: mesh_name(x, mesh_map, fetch=fetch))
    
    if fetch:
        st.sidebar.success("✅ MeSH names updated!")
    
    return result

def search_pubmed_flexible(pubmed_df, drug_name, gene_name=None):
    """More flexible PubMed search that handles partial matches"""
    if pubmed_df.empty:
        return pd.DataFrame()
    
    # Convert to lowercase for case-insensitive matching
    pubmed_lower = pubmed_df.copy()
    pubmed_lower['Abstract_lower'] = pubmed_df['Abstract'].fillna('').str.lower()
    pubmed_lower['Title_lower'] = pubmed_df['Title'].fillna('').str.lower()
    
    drug_lower = drug_name.lower()
    
    # Search in both title and abstract
    drug_in_abstract = pubmed_lower['Abstract_lower'].str.contains(drug_lower, na=False)
    drug_in_title = pubmed_lower['Title_lower'].str.contains(drug_lower, na=False)
    
    drug_mask = drug_in_abstract | drug_in_title
    
    if gene_name:
        gene_lower = str(gene_name).lower()
        gene_in_abstract = pubmed_lower['Abstract_lower'].str.contains(gene_lower, na=False)
        gene_in_title = pubmed_lower['Title_lower'].str.contains(gene_lower, na=False)
        gene_mask = gene_in_abstract | gene_in_title
        combined_mask = drug_mask & gene_mask
    else:
        combined_mask = drug_mask
    
    results = pubmed_df[combined_mask].copy()
    
    # Score relevance
    def compute_relevance(row, drug, gene):
        score = 0
        abstract = row.get('Abstract', '').lower() if pd.notna(row.get('Abstract')) else ''
        title = row.get('Title', '').lower() if pd.notna(row.get('Title')) else ''
        
        # Drug in title gets highest score
        if drug in title:
            score += 3
        if drug in abstract:
            score += 2
        
        if gene:
            if gene in title:
                score += 2
            if gene in abstract:
                score += 1
        
        return score
    
    if not results.empty:
        results['relevance_score'] = results.apply(
            lambda row: compute_relevance(row, drug_lower, gene_lower if gene_name else None), 
            axis=1
        )
        results = results.sort_values('relevance_score', ascending=False)
    
    return results

# ---------------------- INITIALIZE DATA ----------------------
rich, pubmed, mesh_map = load_core()

# ---------------------- INITIALIZE SESSION STATE ----------------------
if 'history' not in st.session_state:
    st.session_state.history = load_json_safe(HISTORY_FILE, [])
if 'pubmed_display_count' not in st.session_state:
    st.session_state.pubmed_display_count = 5
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'drug_input' not in st.session_state:
    st.session_state.drug_input = ""
if 'gene_input' not in st.session_state:
    st.session_state.gene_input = ""

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("🔍 Search Parameters")

# Use session state for inputs so they persist when history is clicked
drug_input = st.sidebar.text_input("Drug Name", value=st.session_state.drug_input, placeholder="e.g., Warfarin").strip()
gene_input = st.sidebar.text_input("Gene Symbol (optional)", value=st.session_state.gene_input, placeholder="e.g., VKORC1").strip()

# Update session state
st.session_state.drug_input = drug_input
st.session_state.gene_input = gene_input

col_btn1, col_btn2 = st.sidebar.columns(2)
run_button = col_btn1.button("⚡ Predict ADR")
clear_button = col_btn2.button("🧹 Clear")

auto_fetch = st.sidebar.checkbox("Auto-fill missing ADR names via NLM", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Recent Searches")

# Display history - fix the click functionality
current_history = st.session_state.history
history_clicked = None

for i, item in enumerate(current_history):
    drug = item.get("drug", "")
    gene = item.get("gene", "")
    label = f"{drug}{f' + {gene}' if gene else ''}"
    if st.sidebar.button(label, key=f"hist_{i}"):
        history_clicked = item

if clear_button:
    st.session_state.history = []
    st.session_state.drug_input = ""
    st.session_state.gene_input = ""
    save_json(HISTORY_FILE, [])
    st.sidebar.success("History cleared!")

# ---------------------- HEADER ----------------------
st.markdown("<h1>DGANet-Rich Clinical Decision Support System</h1>", unsafe_allow_html=True)
st.write("Predict **Adverse Drug Reactions (ADRs)** using literature-informed embeddings.")

# ---------------------- QUERY PROCESSING ----------------------
def process_query(drug_text, gene_text, fetch_mesh=False):
    if not drug_text:
        st.error("❌ Please enter a drug name")
        return None

    # Find matching drug
    unique_drugs = rich["drug"].dropna().unique().tolist()
    matched_drug, score = fuzzy_match_drug(drug_text, unique_drugs)
    
    if matched_drug is None:
        # Show only top 5 suggestions instead of all
        suggestions = difflib.get_close_matches(drug_text, unique_drugs, n=5, cutoff=0.3)
        suggestion_text = f" Try: {', '.join(suggestions)}" if suggestions else ""
        st.error(f"❌ No match found for '{drug_text}'.{suggestion_text}")
        return None
    
    # FIXED: Always show warning when the matched drug is different from input
    if matched_drug.lower() != drug_text.lower():
        st.warning(f"⚠️ Using closest match: '{matched_drug}' (confidence: {score:.2f})")

    # Get ALL data for this drug first
    drug_mask = rich["drug"].str.lower() == matched_drug.lower()
    drug_data = rich[drug_mask].copy()
    
    if drug_data.empty:
        st.error(f"❌ No data found for drug '{matched_drug}'")
        return None

    # Apply gene filter if provided - IMPROVED GENE MATCHING
    gene_text = gene_text.strip()
    filtered_data = drug_data.copy()
    matched_gene = None
    
    if gene_text:
        # Get available genes for this drug
        available_genes = drug_data["gene"].dropna().unique()
        
        # Use fuzzy matching for genes
        matched_gene, gene_score = fuzzy_match_gene(gene_text, available_genes)
        
        if matched_gene is None:
            # Show only top 5 gene suggestions instead of all
            gene_suggestions = difflib.get_close_matches(gene_text, available_genes, n=5, cutoff=0.3)
            suggestion_text = f" Try: {', '.join(map(str, gene_suggestions))}" if gene_suggestions else ""
            st.error(f"❌ Gene '{gene_text}' not found for drug '{matched_drug}'.{suggestion_text}")
            return None
        
        if gene_score < 0.9:
            st.warning(f"🔍 Using closest gene match: '{matched_gene}' (confidence: {gene_score:.2f})")
        
        # Filter by the matched gene
        filtered_data = drug_data[drug_data["gene"] == matched_gene].copy()
        
        if filtered_data.empty:
            st.error(f"❌ No predictions for drug '{matched_drug}' with gene '{matched_gene}'")
            return None
    else:
        # If no gene specified, we still need to pick the best gene for each ADR
        filtered_data = drug_data.copy()

    # Check if we have any non-zero predictions
    max_score = filtered_data["prediction_score"].max()
    if max_score == 0:
        st.warning(f"⚠️ All predictions are zero for '{matched_drug}'{f' with gene {matched_gene}' if matched_gene else ''}. This might indicate limited data.")

    # Group by ADR and get best prediction score for each ADR
    adr_predictions = (filtered_data.sort_values("prediction_score", ascending=False)
                               .groupby("adr", as_index=False)
                               .first()
                               .sort_values("prediction_score", ascending=False))
    
    if adr_predictions.empty:
        st.error("❌ No ADR predictions found after filtering")
        return None

    # Apply MeSH names - FIXED: Now properly using fetch_mesh parameter
    with st.spinner("🔄 Processing ADR names..."):
        adr_predictions["adr_display"] = apply_mesh_names(adr_predictions["adr"], mesh_map, fetch=fetch_mesh)
    
    # Get top predictions
    top_predictions = adr_predictions.head(TOPK_MAIN)
    
    # Get global view (all genes for this drug)
    global_predictions = (drug_data.sort_values("prediction_score", ascending=False)
                                   .groupby("adr", as_index=False)
                                   .first()
                                   .sort_values("prediction_score", ascending=False))
    
    with st.spinner("🔄 Processing global ADR names..."):
        global_predictions["adr_display"] = apply_mesh_names(global_predictions["adr"], mesh_map, fetch=fetch_mesh)
    
    # PubMed results - USING IMPROVED FLEXIBLE SEARCH
    pm_results = search_pubmed_flexible(pubmed, matched_drug, matched_gene if matched_gene else None)
    
    # Summary stats
    mean_score = top_predictions["prediction_score"].mean() if not top_predictions.empty else 0.0
    top_adr = top_predictions.iloc[0]["adr_display"] if not top_predictions.empty else "—"
    
    return {
        "drug": matched_drug,
        "gene": matched_gene if matched_gene else (gene_text if gene_text else None),
        "top_predictions": top_predictions,
        "all_predictions": adr_predictions,
        "global_predictions": global_predictions,
        "pubmed_results": pm_results,
        "summary": {
            "mean_score": mean_score,
            "top_count": len(top_predictions),
            "top_adr": top_adr,
            "max_score": max_score
        }
    }

# ---------------------- EXECUTE QUERY ----------------------
result = None
query_executed = False

# Handle history clicks first
if history_clicked:
    st.session_state.drug_input = history_clicked.get("drug", "")
    st.session_state.gene_input = history_clicked.get("gene", "")
    # Set the flag to run query with history data
    drug_input = st.session_state.drug_input
    gene_input = st.session_state.gene_input
    run_button = True  # This will trigger the query execution below

# Execute query when run button is clicked or history is used
if run_button and drug_input:
    current_query = f"{drug_input}_{gene_input}"
    if current_query != st.session_state.last_query:
        st.session_state.pubmed_display_count = 5
        st.session_state.last_query = current_query
    
    with st.spinner("🔍 Searching for predictions..."):
        result = process_query(drug_input, gene_input, fetch_mesh=auto_fetch)
    
    if result:
        # Update history
        new_entry = {"drug": result["drug"], "gene": result["gene"] or ""}
        history = [entry for entry in st.session_state.history 
                  if not (entry.get("drug") == new_entry["drug"] and entry.get("gene") == new_entry["gene"])]
        history.insert(0, new_entry)
        st.session_state.history = history[:10]  # Keep last 10
        save_json(HISTORY_FILE, st.session_state.history)
        
        st.session_state.current_result = result
        query_executed = True

# Use stored result if available
if st.session_state.current_result and not query_executed:
    result = st.session_state.current_result

# ---------------------- DISPLAY RESULTS ----------------------
if result:
    # Header
    gene_display = result["gene"] if result["gene"] else "All Genes"
    st.markdown(f"""
    <div class='query-card'>
      <b>Query:</b> 🧪 <b>Drug:</b> {result['drug']} | 🧬 <b>Gene:</b> {gene_display}
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Top ADRs Predicted</div>
            <div class='metric-big'>{result['summary']['top_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Mean Prediction Score</div>
            <div class='metric-big'>{result['summary']['mean_score']:.3f}</div>
            <div class='metric-sub'>Top {TOPK_MAIN} ADRs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Highest ADR</div>
            <div class='metric-big'>{result['summary']['top_adr'][:25]}{'...' if len(result['summary']['top_adr']) > 25 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts - Only show if we have non-zero predictions
    st.markdown("<div class='section-title'>📊 ADR Prediction Charts</div>", unsafe_allow_html=True)
    
    if result['summary']['max_score'] > 0:
        chart_col1, chart_col2 = st.columns([1, 1])
        
        # Chart 1: Top ADRs
        with chart_col1:
            st.subheader(f"Top {TOPK_MAIN} ADRs")
            if not result['top_predictions'].empty:
                top_df = result['top_predictions'].head(TOPK_MAIN).copy()
                # Ensure we have some positive values for display
                if top_df['prediction_score'].max() > 0:
                    fig1 = px.bar(
                        top_df.iloc[::-1],
                        x='prediction_score',
                        y='adr_display',
                        orientation='h',
                        color='gene',
                        labels={'prediction_score': 'Prediction Score', 'adr_display': 'ADR'},
                        text=top_df['prediction_score'].round(3)
                    )
                    fig1.update_traces(textposition='outside')
                    fig1.update_layout(height=500, showlegend=True)
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("All prediction scores are zero for this query")
            else:
                st.info("No ADR predictions available")
        
        # Chart 2: Global View
        with chart_col2:
            st.subheader("Global ADR Landscape")
            if not result['global_predictions'].empty:
                global_df = result['global_predictions'].head(GLOBAL_MAX).copy()
                
                # Create scatter plot
                fig2 = px.scatter(
                    global_df,
                    x='adr_display',
                    y='prediction_score',
                    hover_data=['gene', 'adr'],
                    color_discrete_sequence=['lightgray']
                )
                
                # Highlight current selection
                if result['gene']:
                    current_adrs = set(result['top_predictions']['adr'])
                    highlight_mask = global_df['adr'].isin(current_adrs)
                    highlight_df = global_df[highlight_mask]
                    
                    if not highlight_df.empty:
                        fig2.add_scatter(
                            x=highlight_df['adr_display'],
                            y=highlight_df['prediction_score'],
                            mode='markers',
                            marker=dict(size=10, color='blue'),
                            name=f"Gene: {result['gene']}",
                            hovertext=highlight_df['gene']
                        )
                
                fig2.update_layout(
                    height=500,
                    xaxis_tickangle=45,
                    xaxis_title='ADR',
                    yaxis_title='Prediction Score'
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No global ADR data available")
    else:
        st.info("📊 No charts to display - all prediction scores are zero")

    # Detailed Table
    st.markdown("### 📋 Detailed Predictions")
    if not result['all_predictions'].empty:
        display_df = result['all_predictions'][['adr_display', 'gene', 'prediction_score']].rename(
            columns={'adr_display': 'ADR', 'gene': 'Gene', 'prediction_score': 'Score'}
        )
        st.dataframe(
            display_df.style.format({'Score': '{:.3f}'}),
            use_container_width=True,
            height=400
        )
        
        # Show cache status
        mesh_ids_count = display_df['ADR'].str.contains(r'^D\d{6}$', na=False).sum()
        if mesh_ids_count > 0 and auto_fetch:
            st.info(f"🔍 {mesh_ids_count} ADR names still showing as MeSH IDs. The NLM API might be unavailable or the IDs might not exist.")
        elif mesh_ids_count > 0:
            st.info(f"ℹ️ {mesh_ids_count} ADR names showing as MeSH IDs. Check 'Auto-fill missing ADR names via NLM' to fetch human-readable names.")
        
    else:
        st.info("No detailed predictions to display")

    # PubMed Results
    st.markdown("### 📚 Supporting Literature")
    if not result['pubmed_results'].empty:
        pm_df = result['pubmed_results']
        st.write(f"Found **{len(pm_df)}** PubMed relevant articles")
        
        # Display articles with expandable sections
        display_count = st.session_state.pubmed_display_count
        articles_to_show = pm_df.head(display_count)
        
        for idx, row in articles_to_show.iterrows():
            with st.expander(f"📄 {row.get('Title', 'No title')[:80]}...", expanded=False):
                st.write(f"**PMID:** {row.get('PMID', 'N/A')}")
                st.write(f"**Title:** {row.get('Title', 'No title')}")
                st.write(f"**Abstract:** {row.get('Abstract', 'No abstract available')}")
        
        # Show more/show all buttons
        if display_count < len(pm_df):
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("⬇️ Show More", key="show_more"):
                    st.session_state.pubmed_display_count += 5
                    st.rerun()
            
            with col2:
                if st.button("📖 Show All", key="show_all"):
                    st.session_state.pubmed_display_count = len(pm_df)
                    st.rerun()
        
        # Show less button if showing all
        if display_count >= len(pm_df) and len(pm_df) > 5:
            if st.button("⬆️ Show Less", key="show_less"):
                st.session_state.pubmed_display_count = 5
                st.rerun()
                
    else:
        st.info("No PubMed articles found for this query")

# ---------------------- FOOTER ----------------------
st.markdown("""
<div class='footer-note'>
<hr>
💡 <i>This tool is for research purposes only. Always validate predictions with domain experts and clinical evidence.</i>
</div>
""", unsafe_allow_html=True)

# ---------------------- TEAM CREDITS ----------------------
st.markdown("""
<div class='team-credits'>
Developed by: Dakshi Arora, Parag Garg under the mentorship of Dr. Yogesh Gupta | School of Engineering and Technology, BML Munjal University
</div>
""", unsafe_allow_html=True)