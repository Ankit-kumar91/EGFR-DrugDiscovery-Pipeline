"""Streamlit app for the EGFR Drug Discovery Pipeline.

Run with: streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from rdkit import Chem
from rdkit.Chem import Draw

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EGFR Drug Discovery Pipeline",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG_PATH = "configs/config.yaml"


@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()

# ─── Sidebar navigation ──────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "EDA Dashboard",
        "IC50 Predictor",
        "Model Comparison",
        "Virtual Screening Results",
        "Docking Results",
        "MD Analysis",
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1: Project Overview
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview():
    st.title("EGFR Drug Discovery Pipeline")
    st.markdown(
        """
        An end-to-end computational drug discovery pipeline targeting **Epidermal Growth
        Factor Receptor (EGFR)** — a key oncology target.

        ### Pipeline Stages
        ```
        ChEMBL Data → Feature Engineering → ML Training (RF + XGB + GNN)
            → Virtual Screening (Enamine) → Glide Docking → GROMACS MD
        ```

        ### Methodology
        | Stage | Tool / Method |
        |-------|--------------|
        | Data source | ChEMBL (target CHEMBL203) |
        | Features | Morgan fingerprints (ECFP4) + RDKit descriptors |
        | Classical ML | Random Forest, XGBoost |
        | Deep learning | GCN-based GNN (PyTorch Geometric) |
        | Experiment tracking | MLflow |
        | Virtual screening | Enamine Diversity Set |
        | Docking | Schrödinger Glide (SP) |
        | MD simulation | GROMACS (100 ns, AMBER99SB-ILDN) |
        """
    )

    # Dataset stats
    curated_path = "data/processed/egfr_curated.csv"
    if os.path.exists(curated_path):
        df = pd.read_csv(curated_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Compounds", len(df))
        col2.metric("Active (pIC50 ≥ 6)", (df["activity_class"] == "active").sum())
        col3.metric("Inactive", (df["activity_class"] == "inactive").sum())
        col4.metric("Mean pIC50", f"{df['pIC50'].mean():.2f}")
    else:
        st.info("Run the data curation pipeline first to see dataset statistics.")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2: EDA Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def page_eda():
    st.title("Exploratory Data Analysis")
    curated_path = "data/processed/egfr_curated.csv"

    if not os.path.exists(curated_path):
        st.warning("Curated dataset not found. Run `python main.py --stage data` first.")
        return

    df = pd.read_csv(curated_path)

    tab1, tab2, tab3 = st.tabs(["Distribution", "Chemical Space", "Descriptors"])

    with tab1:
        st.subheader("pIC50 Distribution")
        import plotly.express as px

        fig = px.histogram(
            df, x="pIC50", color="activity_class", nbins=50, barmode="overlay",
            color_discrete_map={"active": "#2ecc71", "inactive": "#e74c3c"},
            title="pIC50 Distribution by Activity Class",
        )
        fig.add_vline(x=6.0, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df["pIC50"].describe().to_frame("pIC50 Stats"))
        with col2:
            st.dataframe(df["activity_class"].value_counts().to_frame("Count"))

    with tab2:
        plot_path = "results/plots/chemical_space.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="PCA & t-SNE of Morgan Fingerprints")
        else:
            st.info("Run notebook 02 to generate chemical space plots.")

    with tab3:
        plot_path = "results/plots/descriptor_distributions.png"
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Descriptor Distributions")
        else:
            st.info("Run notebook 02 to generate descriptor plots.")

        corr_path = "results/plots/descriptor_correlation.png"
        if os.path.exists(corr_path):
            st.image(corr_path, caption="Descriptor Correlation Matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3: IC50 Predictor
# ═══════════════════════════════════════════════════════════════════════════════
def page_predictor():
    st.title("IC50 Predictor")
    st.markdown("Enter a SMILES string to predict its pIC50 against EGFR.")

    smiles_input = st.text_input(
        "SMILES",
        value="c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
        help="Enter a valid SMILES string",
    )

    if st.button("Predict", type="primary"):
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES string. Please check your input.")
            return

        # Show 2D structure
        col1, col2 = st.columns([1, 2])
        with col1:
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="2D Structure")

        with col2:
            try:
                from src.pipeline.predict_pipeline import PredictPipeline
                pipeline = PredictPipeline()
                result = pipeline.predict_single(smiles_input)

                # Prediction results
                st.subheader("Predicted pIC50")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Random Forest", f"{result['pIC50_RF']:.3f}")
                m2.metric("XGBoost", f"{result['pIC50_XGB']:.3f}")
                m3.metric("GNN", f"{result['pIC50_GNN']:.3f}" if result["pIC50_GNN"] else "N/A")
                m4.metric("Consensus", f"{result['pIC50_consensus']:.3f}")

                # Activity prediction
                if result["activity"] == "active":
                    st.success(f"Predicted: **ACTIVE** (pIC50 = {result['pIC50_consensus']:.3f})")
                else:
                    st.warning(f"Predicted: **INACTIVE** (pIC50 = {result['pIC50_consensus']:.3f})")

                # Molecular properties
                st.subheader("Molecular Properties")
                props = {
                    k: v for k, v in result.items()
                    if k in ["MolWt", "LogP", "NumHDonors", "NumHAcceptors",
                             "Lipinski_Violations", "Lipinski_Pass"]
                }
                st.json(props)

            except Exception as e:
                st.error(
                    f"Prediction failed: {e}\n\n"
                    "Ensure models are trained first: `python main.py --stage train`"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4: Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def page_model_comparison():
    st.title("Model Comparison")

    comparison_path = "results/model_comparison.csv"
    if not os.path.exists(comparison_path):
        st.warning("No model comparison data. Run training first.")
        return

    comparison = pd.read_csv(comparison_path, index_col="Model")
    st.dataframe(comparison.style.highlight_max(axis=0, subset=["R2", "Pearson_r"])
                 .highlight_min(axis=0, subset=["RMSE", "MAE"]))

    # Plots
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance"])

    with tab1:
        for model in comparison.index:
            path = f"results/plots/{model}_actual_vs_predicted.png"
            if os.path.exists(path):
                st.image(path, caption=f"{model}: Actual vs Predicted")

    with tab2:
        for model in comparison.index:
            path = f"results/plots/{model}_residuals.png"
            if os.path.exists(path):
                st.image(path, caption=f"{model}: Residuals")

    with tab3:
        for model in ["RandomForest", "XGBoost"]:
            path = f"results/plots/{model}_feature_importance.png"
            if os.path.exists(path):
                st.image(path, caption=f"{model}: Feature Importance")

    # Learning curves
    st.subheader("Learning Curves")
    for model in ["RandomForest", "XGBoost"]:
        path = f"results/plots/{model}_learning_curves.png"
        if os.path.exists(path):
            st.image(path, caption=f"{model}: Learning Curves")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 5: Virtual Screening Results
# ═══════════════════════════════════════════════════════════════════════════════
def page_virtual_screening():
    st.title("Virtual Screening Results")

    hits_path = "results/virtual_screening_hits.csv"
    if not os.path.exists(hits_path):
        st.warning("No virtual screening results. Run `python main.py --stage screen` first.")
        return

    hits = pd.read_csv(hits_path)
    st.metric("Total Hits", len(hits))

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_pIC50 = st.slider("Minimum predicted pIC50", 5.0, 10.0, 6.0, 0.1)
    with col2:
        lipinski_only = st.checkbox("Lipinski-compliant only", value=True)

    filtered = hits[hits["predicted_pIC50"] >= min_pIC50]
    if lipinski_only and "Lipinski_Pass" in filtered.columns:
        filtered = filtered[filtered["Lipinski_Pass"]]

    st.dataframe(filtered, use_container_width=True)

    # Chemical space plot
    overlap_path = "results/plots/vs_chemical_space_overlap.png"
    if os.path.exists(overlap_path):
        st.image(overlap_path, caption="Chemical Space: Training Set vs Hits")

    # 2D structures of top hits
    if len(filtered) > 0 and "canonical_smiles" in filtered.columns:
        st.subheader("Top Hit Structures")
        top_n = min(10, len(filtered))
        mols = []
        legends = []
        for _, row in filtered.head(top_n).iterrows():
            mol = Chem.MolFromSmiles(row["canonical_smiles"])
            if mol:
                mols.append(mol)
                legends.append(f"pIC50={row['predicted_pIC50']:.2f}")
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(250, 250), legends=legends)
            st.image(img)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 6: Docking Results
# ═══════════════════════════════════════════════════════════════════════════════
def page_docking():
    st.title("Molecular Docking Results (Glide)")

    glide_path = "results/docking/glide_results.csv"
    top_poses_path = "results/docking/top_poses/top_docked_compounds.csv"

    if os.path.exists(glide_path):
        results = pd.read_csv(glide_path)
        st.metric("Docked Compounds", len(results))
        st.dataframe(results, use_container_width=True)

        if "GlideScore" in results.columns:
            import plotly.express as px
            fig = px.histogram(results, x="GlideScore", nbins=30,
                               title="GlideScore Distribution")
            st.plotly_chart(fig, use_container_width=True)
    elif os.path.exists(top_poses_path):
        top = pd.read_csv(top_poses_path)
        st.dataframe(top, use_container_width=True)
    else:
        st.info(
            "No docking results available. Run `python main.py --stage dock` "
            "with Schrödinger Glide installed."
        )

    # Binding interaction images
    st.subheader("Binding Interactions")
    for i in range(1, 3):
        img_path = f"results/docking/top_poses/compound_{i}_interactions.png"
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Compound {i} Binding Interactions")

    score_path = "results/plots/glide_score_distribution.png"
    if os.path.exists(score_path):
        st.image(score_path, caption="GlideScore Distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# Page 7: MD Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def page_md_analysis():
    st.title("MD Simulation Analysis")

    st.markdown(
        f"""
        **Protocol:** {config['md_simulation']['force_field']} / {config['md_simulation']['water_model']}
        | **Time:** {config['md_simulation']['simulation_time_ns']} ns
        | **Temperature:** {config['md_simulation']['temperature_k']} K
        """
    )

    tab1, tab2 = st.tabs(["Compound 1", "Compound 2"])

    for tab, compound in zip([tab1, tab2], ["compound_1", "compound_2"]):
        with tab:
            base = f"results/md_simulation/{compound}"

            # RMSD
            rmsd_path = f"{base}/rmsd.png"
            if os.path.exists(rmsd_path):
                st.image(rmsd_path, caption="RMSD over Time")
            else:
                st.info("RMSD plot not available. Run MD simulation first.")

            # RMSF
            rmsf_path = f"{base}/rmsf.png"
            if os.path.exists(rmsf_path):
                st.image(rmsf_path, caption="RMSF per Residue")

            # Radius of gyration
            rg_path = f"{base}/rg.png"
            if os.path.exists(rg_path):
                st.image(rg_path, caption="Radius of Gyration")

            # Check for MDP files
            mdp_files = [f for f in os.listdir(base) if f.endswith(".mdp")] if os.path.exists(base) else []
            if mdp_files:
                st.subheader("Simulation Parameters")
                for mdp in sorted(mdp_files):
                    with st.expander(mdp):
                        with open(os.path.join(base, mdp)) as f:
                            st.code(f.read(), language="ini")

    # Comparison plot
    comparison_path = "results/plots/md_comparison.png"
    if os.path.exists(comparison_path):
        st.subheader("Compound Comparison")
        st.image(comparison_path, caption="RMSD / RMSF / Rg Comparison")


# ─── Page router ──────────────────────────────────────────────────────────────
PAGES = {
    "Project Overview": page_overview,
    "EDA Dashboard": page_eda,
    "IC50 Predictor": page_predictor,
    "Model Comparison": page_model_comparison,
    "Virtual Screening Results": page_virtual_screening,
    "Docking Results": page_docking,
    "MD Analysis": page_md_analysis,
}

PAGES[page]()

# ─── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**EGFR Drug Discovery Pipeline**  \n"
    "Built with MLflow, PyTorch Geometric, RDKit, Streamlit"
)
