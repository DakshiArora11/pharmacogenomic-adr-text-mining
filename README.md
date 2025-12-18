# ✨ Pharmacogenomic ADR Prediction via Text Mining

This repository contains the **codebase** for a pharmacogenomics research project focused on **predicting adverse drug reactions (ADRs)** by integrating **structured biomedical knowledge** with **unstructured evidence mined from biomedical literature**.

The project extends the DGANet framework by incorporating **literature-derived signals**, enabling more comprehensive, explainable, and evidence-backed **drug–gene–ADR association modeling**.

---

## 🔍 Problem Motivation

Adverse drug reactions often arise due to **genetic variability** in patients. Existing ADR prediction systems predominantly rely on **curated structured resources**, which:

- Miss emerging or rare drug–gene–ADR relationships  
- Underutilize the rapidly growing biomedical literature  
- Provide limited transparency for downstream clinical interpretation  

This work addresses these gaps by **systematically mining unstructured literature** and fusing it with pharmacogenomic knowledge representations.

---

## 🧠 Core Contributions

- **Literature-augmented ADR prediction**  
  Integrates evidence extracted from biomedical literature with structured pharmacogenomic knowledge to strengthen ADR prediction.

- **Knowledge graph–based modeling**  
  Constructs heterogeneous graphs linking drugs, genes, diseases, and adverse reactions for relational learning.

- **Ontology-aware normalization**  
  Employs biomedical ontologies to standardize terminology and enable semantic similarity across entities.

- **Extension of DGANet-style architectures**  
  Enhances graph-based learning with literature-informed feature representations.

- **Explainability-oriented design**  
  Supports evidence tracing, enabling predictions to be linked back to supporting biomedical context.

---

## 🏥 Clinical Relevance

The pipeline is designed as an **end-to-end research framework** that aligns with the requirements of **clinical decision support systems**, enabling:

- Evidence-backed assessment of potential adverse drug reactions  
- Improved interpretability of model predictions for clinical analysis  
- Integration of heterogeneous biomedical knowledge sources  

While intended for **research and development**, the framework is structured to support translational exploration toward clinical risk assessment and pharmacogenomic decision support.

---

