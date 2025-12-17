import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import difflib, json, os, re

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="DGANet Clinical Decision Support System", page_icon="🧠", layout="wide")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #f8fbff, #e8f0fe);}
h1 {text-align:center;font-weight:700;background:-webkit-linear-gradient(45deg,#1565c0,#42a5f5);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5em;}
.section-title {font-size:1.25rem;font-weight:600;margin-top:2rem;color:#0d47a1;border-bottom:2px solid #bbdefb;padding-bottom:0.2rem;}
.query-card {background:#fff;padding:1rem 1.2rem;border-left:5px solid #2196f3;border-radius:10px;
             box-shadow:0 2px 6px rgba(0,0,0,0.1);margin-bottom:1rem;}
.footer-note{text-align:center;color:#666;margin-top:3rem;font-size:0.85rem;}
.history-pill{display:inline-block;background:#e3f2fd;color:#1565c0;padding:6px 12px;margin:4px;
              border-radius:20px;cursor:pointer;font-size:0.9rem;font-weight:500;}
.history-pill:hover{background:#bbdefb;}
</style>
""", unsafe_allow_html=True)

# ---------------------- LOCAL HISTORY ----------------------
HISTORY_FILE = "search_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[:20], f)

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    embeddings = np.load("data/dganet_literature_embeddings.npy")
    nodes = pd.read_csv("data/dga_nodes.csv")
    triples = pd.read_csv("data/triples_normalized.csv")
    pubmed = pd.read_csv("data/pubmed_results_large.csv")
    return embeddings, nodes, triples, pubmed

embeddings, nodes, triples, pubmed = load_data()

# ---------------------- SESSION STATE ----------------------
st.session_state.setdefault("history", load_history())
st.session_state.setdefault("results", None)
st.session_state.setdefault("current_query", None)
st.session_state.setdefault("pubmed_offset", 5)
st.session_state.setdefault("show_all_pubmed", False)

# ---------------------- HEADER ----------------------
st.markdown("<h1>DGANet Clinical Decision Support System</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;font-size:1.05rem;color:#333'>
Predict <b>Adverse Drug Reactions (ADRs)</b> using literature-augmented <b>DGANet embeddings</b>.<br>
Provide a <b>drug</b> (and optionally a <b>gene</b>) to explore predicted risk scores and PubMed evidence.
</p>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("🔍 Search Parameters")

drug_input = st.sidebar.text_input("Drug Name", placeholder="Enter here").strip().lower()
gene_input = st.sidebar.text_input("Gene Symbol (optional)", placeholder="Enter here").strip().upper()

col_btn1, col_btn2 = st.sidebar.columns(2)
run_button = col_btn1.button("⚡ Predict ADR")
clear_button = col_btn2.button("🧹 Clear History")

# --- show history ---
st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Recent Searches")
selected_history = None
if st.session_state["history"]:
    for idx, (d, g) in enumerate(st.session_state["history"]):
        label = f"{d} ({g})" if g else d
        if st.sidebar.button(label, key=f"hist_{idx}"):
            selected_history = (d, g)
else:
    st.sidebar.info("No searches yet.")

if clear_button:
    st.session_state["history"].clear()
    save_history([])
    st.sidebar.success("History cleared.")

# ---------------------- MATCHING ----------------------
def normalize(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())

def smart_match(drug_input, nodes, min_confidence=0.8):
    """
    Try to find the most relevant node for a drug.
    If fuzzy match similarity < min_confidence → reject as irrelevant.
    """
    nodes["clean_name"] = (
        nodes["Node"]
        .str.replace("DRUG_", "")
        .str.replace("GENE_", "")
        .str.replace("ADR_", "")
        .str.lower()
    )

    # Direct partial match first
    partial = nodes[nodes["clean_name"].str.contains(drug_input, na=False)]
    if not partial.empty:
        return partial.iloc[0], None, 1.0  # perfect confidence

    # Fuzzy match fallback
    all_names = nodes["clean_name"].dropna().tolist()
    matches = difflib.get_close_matches(drug_input, all_names, n=1, cutoff=0.3)
    if matches:
        match = matches[0]
        score = difflib.SequenceMatcher(None, drug_input, match).ratio()
        node_row = nodes[nodes["clean_name"] == match].iloc[0]
        return node_row, match, score

    return None, None, 0.0


# ---------------------- MAIN QUERY ----------------------
def run_query(drug_input, gene_input):
    node_row, fuzzy_match, match_confidence = smart_match(drug_input, nodes)

    if node_row is None:
        st.error(f"❌ No matching node found for '{drug_input}'. Please try another name.")
        return None, None, None, drug_input

    # Handle fuzzy match relevance
    corrected_name = drug_input
    if fuzzy_match:
        if match_confidence < 0.8:
            st.error(
                f"❌ No relevant match found for '{drug_input}'. "
                "Please try again with a more specific or correct drug name."
            )
            return None, None, None, drug_input
        else:
            st.info(f"🔎 Fuzzy matched '{drug_input}' → **{fuzzy_match}** (confidence: {match_confidence:.2f})")
            corrected_name = fuzzy_match

    found_node = node_row["Node"]
    drug_emb = embeddings[node_row.name].reshape(1, -1)

    adr_nodes = nodes[nodes["Node"].str.startswith("ADR_")]
    adr_embs = embeddings[adr_nodes.index]
    sims = cosine_similarity(drug_emb, adr_embs)[0]

    results = pd.DataFrame({
        "ADR_Node": adr_nodes["Node"].values,
        "Similarity Score": sims
    }).sort_values("Similarity Score", ascending=False).head(10)

    matches = pubmed[pubmed["Abstract"].str.contains(corrected_name, case=False, na=False)]
    if gene_input:
        matches = matches[matches["Abstract"].str.contains(gene_input, case=False, na=False)]

    return found_node, results, matches, corrected_name

# ---------------------- RUN LOGIC ----------------------
if run_button and drug_input:
    with st.spinner("🔬 Running prediction... please wait"):
        found_node, results, matches, corrected_name = run_query(drug_input, gene_input)
        if results is not None:
            # ✅ store corrected name for display + history
            st.session_state["results"] = (found_node, results, matches)
            st.session_state["current_query"] = (corrected_name, gene_input)
            st.session_state["pubmed_offset"] = 5
            st.session_state["show_all_pubmed"] = False
            if (corrected_name, gene_input) not in st.session_state["history"]:
                st.session_state["history"].insert(0, (corrected_name, gene_input))
                save_history(st.session_state["history"])

elif selected_history:
    drug_input, gene_input = selected_history
    with st.spinner("🔁 Loading from history..."):
        found_node, results, matches, corrected_name = run_query(drug_input, gene_input)
        st.session_state["results"] = (found_node, results, matches)
        st.session_state["current_query"] = (corrected_name, gene_input)
        st.session_state["pubmed_offset"] = 5
        st.session_state["show_all_pubmed"] = False

# ---------------------- DISPLAY ----------------------
if st.session_state["results"]:
    found_node, results, matches = st.session_state["results"]
    drug, gene = st.session_state["current_query"]

    st.markdown(f"""
    <div class='query-card'>
      <b>Showing Results for:</b><br>
      🧪 <b>Drug:</b> {drug.title()}<br>
      {f"🧬 <b>Gene:</b> {gene}" if gene else ""}
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.55, 0.45])
    with col1:
        st.markdown("<div class='section-title'>📊 Top Predicted ADR Risks</div>", unsafe_allow_html=True)
        st.dataframe(results.style.format({"Similarity Score": "{:.3f}"}))

    with col2:
        st.markdown("<div class='section-title'>📚 Supporting PubMed Literature</div>", unsafe_allow_html=True)
        if matches is not None and not matches.empty:
            start = 0
            end = st.session_state["pubmed_offset"]
            visible = matches.iloc[start:end]

            for _, row in visible.iterrows():
                title = row.get("Title", "")
                pmid = row.get("PMID", "N/A")
                abs_text = row["Abstract"]
                with st.expander(f"🩺 PMID {pmid} — *{title[:100]}...*"):
                    st.markdown(f"**PMID:** {pmid}")
                    st.markdown(f"**Title:** {title}")
                    st.markdown(f"**Abstract:** {abs_text}")

            total = len(matches)
            if not st.session_state["show_all_pubmed"]:
                col_more, col_all = st.columns([0.5, 0.5])
                if end < total and col_more.button("⬇️ Show More Results"):
                    st.session_state["pubmed_offset"] += 5
                    st.rerun()
                if col_all.button("📖 Show All Results"):
                    st.session_state["show_all_pubmed"] = True
                    st.rerun()
            else:
                st.info(f"Showing all {total} PubMed results.")
                for _, row in matches.iloc[end:].iterrows():
                    title = row.get("Title", "")
                    pmid = row.get("PMID", "N/A")
                    abs_text = row["Abstract"]
                    with st.expander(f"🩺 PMID {pmid} — *{title[:100]}...*"):
                        st.markdown(f"**PMID:** {pmid}")
                        st.markdown(f"**Title:** {title}")
                        st.markdown(f"**Abstract:** {abs_text}")
        else:
            st.info("No supporting PubMed evidence found for this query.")

# ---------------------- FOOTER ----------------------
st.markdown("""
<div class='footer-note'>
<hr>
💡 <i>This tool is for research purposes only. Predictions should be validated by domain experts.</i>
</div>
""", unsafe_allow_html=True)
