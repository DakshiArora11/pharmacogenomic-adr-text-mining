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

# Paths (change if yours differ)
FEATURES_PATH = os.getenv("DGANET_FEATURES_PATH", r"D:/dganet_features/features_rich_full.csv.gz")
PRED_PATH     = os.getenv("DGANET_PREDICTIONS_PATH", r"D:/dganet_features/predictions_rich.csv")
PUBMED_PATH   = "data/pubmed_results_large.csv"
MESH_CACHE    = "data/umls_mesh_cache.json"
HISTORY_FILE  = "search_history.json"

TOPK_MAIN = 20         # how many ADRs to show in the top-chart and cards
GLOBAL_MAX = 60        # how many ADRs to show on the scatter for readability

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
.history-pill{display:inline-block;background:#e3f2fd;color:#1565c0;padding:6px 12px;margin:4px;border-radius:20px;cursor:pointer;font-size:0.9rem;font-weight:500;}
.history-pill:hover{background:#bbdefb;}
.metric-card {background:#fff;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.08);padding:18px;text-align:center;}
.metric-title {color:#607d8b;font-size:0.85rem;margin-bottom:6px;}
.metric-big {font-size:1.75rem;font-weight:800;color:#0d47a1;}
.metric-sub {font-size:0.85rem;color:#607d8b;}
.smallnote {font-size:0.8rem;color:#607d8b;}
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

def normalize(s): return re.sub(r"[^a-z0-9]+", "", str(s).lower())

@st.cache_data(show_spinner=False)
def load_core():
    # features (ids & labels) + predictions
    rich = pd.read_csv(FEATURES_PATH, compression="gzip")
    preds = pd.read_csv(PRED_PATH)
    if "prediction_score" not in preds.columns:
        raise RuntimeError("predictions_rich.csv must have column 'prediction_score'")
    if len(preds) != len(rich):
        st.warning("Length mismatch between FEATURES and PREDICTIONS. Merging by row order.")
    rich = rich.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    rich["prediction_score"] = preds["prediction_score"]
    # PubMed
    pubmed = pd.read_csv(PUBMED_PATH)
    # Cache for MeSH names
    mesh_map = load_json_safe(MESH_CACHE, {})
    return rich, pubmed, mesh_map

def fuzzy_match_drug(drug_text, drug_list):
    # exact normalized, then difflib fallback
    dnorm = normalize(drug_text)
    for d in drug_list:
        if normalize(d) == dnorm:
            return d, 1.0
    best = difflib.get_close_matches(drug_text, drug_list, n=1, cutoff=0.6)
    if best:
        score = difflib.SequenceMatcher(None, drug_text.lower(), best[0].lower()).ratio()
        return best[0], score
    return None, 0.0

def mesh_name(code, mesh_map, fetch=False):
    """Map ADR code/text to pretty name. If fetch=True and looks like MeSH ID, try NLM API once and persist."""
    if not isinstance(code, str):
        return str(code)
    # Already a readable name (no D[0-9] code)
    if not re.fullmatch(r"D\d{6}", code):
        return code.replace("_", " ").replace("-", " ").title()
    if code in mesh_map and mesh_map[code]:
        return mesh_map[code]
    if not fetch:
        return code  # fallback: leave as code
    # try NLM once
    try:
        url = f"https://id.nlm.nih.gov/mesh/lookup/label?label={code}&match=exact"
        r = requests.get(url, timeout=4)
        if r.ok:
            data = r.json()
            # the API returns entries with 'label'; sometimes empty for exact code query, so second try via record
            if data and "label" in data[0]:
                label = data[0]["label"]
            else:
                # record endpoint
                r2 = requests.get(f"https://id.nlm.nih.gov/mesh/{code}.json", timeout=4)
                if r2.ok:
                    j = r2.json()
                    label = next((x.get("label") for x in j if x.get("@type") == "Descriptor"), code)
                else:
                    label = code
            mesh_map[code] = label
            save_json(MESH_CACHE, mesh_map)
            return label
    except Exception:
        pass
    return code

def apply_mesh_names(series, mesh_map, fetch=False):
    return series.apply(lambda x: mesh_name(x, mesh_map, fetch=fetch))

def load_history():
    return load_json_safe(HISTORY_FILE, [])

def save_history(h): save_json(HISTORY_FILE, h[:20])

# ---------------------- DATA ----------------------
rich, pubmed, mesh_map = load_core()

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("🔍 Search Parameters")

drug_input = st.sidebar.text_input("Drug Name", placeholder="Enter here").strip()
gene_input = st.sidebar.text_input("Gene Symbol (optional)", placeholder="Enter here").strip()
col_btn1, col_btn2 = st.sidebar.columns(2)
run_button = col_btn1.button("⚡ Predict ADR")
clear_button = col_btn2.button("🧹 Clear History")

st.sidebar.markdown("---")
auto_fetch = st.sidebar.checkbox("Auto-fill missing ADR names via NLM (persists)", value=False)
st.sidebar.markdown("<span class='smallnote'>Uses NLM MeSH API once per missing code and stores it in data/umls_mesh_cache.json.</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Recent Searches")
history = st.session_state.get("history", load_history())
clicked = None
for i, item in enumerate(history):
    d, g = item.get("drug",""), item.get("gene","")
    label = f"{d}{' ('+g+')' if g else ''}"
    if st.sidebar.button(label, key=f"h_{i}"):
        clicked = item

if clear_button:
    history = []
    save_history(history)
    st.session_state["history"] = history
    st.sidebar.success("History cleared.")

# ---------------------- HEADER ----------------------
st.markdown("<h1>DGANet-Rich Clinical Decision Support System</h1>", unsafe_allow_html=True)
st.write("Predict **Adverse Drug Reactions (ADRs)** using literature-informed embeddings. "
         "Enter a **Drug** and optionally a **Gene** to see model predictions, supporting PubMed evidence, and ADR relationships.")

# ---------------------- CORE QUERY ----------------------
def run_query(drug_text, gene_text, fetch_mesh=False):
    if not drug_text:
        return None

    # Smart drug match against what exists in the file
    unique_drugs = sorted(rich["drug"].dropna().unique().tolist())
    matched_drug, score = fuzzy_match_drug(drug_text, unique_drugs)
    if matched_drug is None or score < 0.75:
        st.error(f"❌ No relevant match found for '{drug_text}'. Try another spelling.")
        return None

    df = rich[rich["drug"].str.lower() == matched_drug.lower()].copy()

    # If gene specified, filter to that gene only
    gene_text = gene_text.strip()
    if gene_text:
        df = df[df["gene"].str.upper() == gene_text.upper()].copy()
        if df.empty:
            st.warning(f"No results for **{matched_drug}** with gene **{gene_text}**. Showing nothing.")
            return {
                "drug": matched_drug, "gene": gene_text, "top": pd.DataFrame(), "all_for_drug": pd.DataFrame(),
                "pubmed": pd.DataFrame(), "summary": {"mean": 0.0, "topk": 0, "top_adr": "—"}
            }

    # Compute per-ADR best score (+carry gene for that best)
    # If gene is fixed, this is just the gene’s rows. If not, pick the gene giving max score for each ADR.
    grp = df.sort_values("prediction_score", ascending=False).groupby("adr", as_index=False).first()
    grp = grp.sort_values("prediction_score", ascending=False)

    # Decorate ADR names
    grp["adr_display"] = apply_mesh_names(grp["adr"], mesh_map, fetch=fetch_mesh)

    # Summary cards
    top_slice = grp.head(TOPK_MAIN)
    mean_score = float(top_slice["prediction_score"].mean()) if not top_slice.empty else 0.0
    top_adr = top_slice.iloc[0]["adr_display"] if not top_slice.empty else "—"

    # PubMed
    pm = pubmed[pubmed["Abstract"].str.contains(matched_drug, case=False, na=False)].copy()
    if gene_text:
        pm = pm[pm["Abstract"].str.contains(gene_text, case=False, na=False)]
    # Attach
    return {
        "drug": matched_drug,
        "gene": gene_text if gene_text else None,
        "top": top_slice,
        "all_for_drug": grp,   # all ADRs for this drug (best per ADR)
        "pubmed": pm,
        "summary": {"mean": mean_score, "topk": int(len(top_slice)), "top_adr": top_adr}
    }

# ---------------------- RUN ----------------------
result = None
if run_button and drug_input:
    result = run_query(drug_input, gene_input, fetch_mesh=auto_fetch)
    if result:
        # update history (dedupe/most recent first)
        entry = {"drug": result["drug"], "gene": result["gene"] or ""}
        history = [e for e in history if not (e.get("drug")==entry["drug"] and e.get("gene")==entry["gene"])]
        history.insert(0, entry)
        save_history(history)
        st.session_state["history"] = history

elif clicked:
    result = run_query(clicked["drug"], clicked.get("gene",""), fetch_mesh=auto_fetch)

# ---------------------- DISPLAY ----------------------
if result:
    st.markdown(f"""
    <div class='query-card'>
      <b>Prediction Summary:</b><br>
      🧪 <b>Drug:</b> {result['drug'].title()} {(' | 🧬 <b>Gene:</b> ' + result['gene']) if result['gene'] else ''}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><div class='metric-title'>Top ADRs Predicted</div>"
                    f"<div class='metric-big'>{result['summary']['topk']}</div>"
                    "<div class='metric-sub'>&nbsp;</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='metric-title'>Mean Prediction Score</div>"
                    f"<div class='metric-big'>{result['summary']['mean']:.3f}</div>"
                    "<div class='metric-sub'>of the top set</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='metric-title'>Highest ADR Predicted</div>"
                    f"<div class='metric-big'>{result['summary']['top_adr']}</div>"
                    "<div class='metric-sub'>&nbsp;</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 ADR Prediction Visuals</div>", unsafe_allow_html=True)
    g1, g2 = st.columns([1.05, 1.0])

    # ---------- Chart 1: Top-K ADRs (filtered by gene if given) ----------
    topdf = result["top"].copy()
    if topdf.empty:
        g1.info("No ADRs to show.")
    else:
        topdf = topdf.head(TOPK_MAIN)
        # show ADR name; if user gave a gene, all bars share that gene; else show gene that maximizes each ADR
        topdf["label"] = topdf["adr_display"]
        fig = px.bar(
            topdf.iloc[::-1],
            x="prediction_score", y="label", color="gene",
            orientation="h",
            labels={"prediction_score": "Prediction Score", "label": "ADR", "gene": "Gene"},
            text=topdf.iloc[::-1]["prediction_score"].round(3)
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), legend_title_text="gene")
        g1.plotly_chart(fig, use_container_width=True)

    # ---------- Chart 2: Global ADR Landscape for this drug (best gene per ADR) ----------
    alldf = result["all_for_drug"].copy()
    if alldf.empty:
        g2.info("No ADRs to show.")
    else:
        # limit to avoid unreadable x; keep highest N
        showN = min(GLOBAL_MAX, len(alldf))
        land = alldf.head(showN).copy()
        land["x_label"] = land["adr_display"]
        fig2 = px.scatter(
            land, x="x_label", y="prediction_score",
            hover_data={"gene": True, "adr": True, "x_label": False},
            color_discrete_sequence=["#90a4ae"]
        )
        # highlight mean+3 highest points (blue)
        if showN >= 3:
            hi = land.nlargest(3, "prediction_score").index
            fig2.add_scatter(x=land.loc[hi,"x_label"], y=land.loc[hi,"prediction_score"],
                             mode="markers", marker=dict(size=9), name="Selected Drug",
                             hovertext=land.loc[hi,"gene"])
        fig2.update_layout(
            height=520, margin=dict(l=10,r=10,t=40,b=120),
            xaxis_title="ADR", yaxis_title="Prediction Score"
        )
        fig2.update_xaxes(tickangle=60)
        g2.plotly_chart(fig2, use_container_width=True)

    # ---------- Detailed table ----------
    st.markdown("### 📋 Detailed ADR Predictions")
    table_df = result["all_for_drug"][["adr_display","gene","prediction_score"]].rename(
        columns={"adr_display":"ADR","gene":"Gene","prediction_score":"Prediction Score"}
    )
    st.dataframe(table_df.style.format({"Prediction Score":"{:.3f}"}), use_container_width=True, height=420)

    # ---------- PubMed Section (with Show more / Show all) ----------
    st.markdown("### 📚 Supporting PubMed Literature")
    pm = result["pubmed"].copy()
    if pm.empty:
        st.info("No PubMed hits for this query.")
    else:
        # keep the old “Show more / Show all” behaviour
        offset = st.session_state.get("pubmed_offset", 5)
        show_all = st.session_state.get("show_all_pubmed", False)

        def render_rows(frame):
            for _, row in frame.iterrows():
                title = row.get("Title","") or ""
                pmid = row.get("PMID","N/A")
                abs_text = row.get("Abstract","")
                with st.expander(f"🩺 PMID {pmid} — *{title[:100]}...*"):
                    st.markdown(f"**PMID:** {pmid}")
                    st.markdown(f"**Title:** {title}")
                    st.markdown(f"**Abstract:** {abs_text}")

        if not show_all:
            render_rows(pm.iloc[:offset])
            c_more, c_all = st.columns(2)
            if offset < len(pm) and c_more.button("⬇️ Show More Results"):
                st.session_state["pubmed_offset"] = offset + 5
                st.rerun()
            if c_all.button("📖 Show All Results"):
                st.session_state["show_all_pubmed"] = True
                st.rerun()
        else:
            st.info(f"Showing all {len(pm)} PubMed results.")
            render_rows(pm)

# ---------------------- FOOTER ----------------------
st.markdown("<div class='footer-note'><hr>💡 <i>This tool is for research purposes only. Validate with domain experts.</i></div>", unsafe_allow_html=True)
