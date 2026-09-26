import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as io_go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Page Config
st.set_page_config(
    page_title="Evolutionary Customer Segmentation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Aesthetic & KPI Cards
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.5);
    }

    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F9FAFB;
        margin-top: 0.3rem;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #10B981;
        margin-top: 0.2rem;
    }

    .persona-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "segmented_customers.csv")


@st.cache_data
def load_data(file_path: str):
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df


@st.cache_data
def compute_pca(df: pd.DataFrame):
    feature_cols = ["Recency", "Frequency", "Monetary", "AOV"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    pca_df["Customer ID"] = df["Customer ID"]
    pca_df["Cluster"] = df["Cluster"].astype(str)
    pca_df["Recency"] = df["Recency"]
    pca_df["Frequency"] = df["Frequency"]
    pca_df["Monetary"] = df["Monetary"]
    pca_df["AOV"] = df["AOV"]

    exp_var = pca.explained_variance_ratio_
    return pca_df, exp_var


# Cluster Persona Definitions
CLUSTER_PERSONAS = {
    0: {
        "name": "VIP High Spenders",
        "color": "#8B5CF6",  # Purple
        "badge_bg": "linear-gradient(135deg, #8B5CF6, #C084FC)",
        "description": "High total monetary spend and frequent orders. Essential revenue drivers.",
        "strategy": "Offer exclusive VIP loyalty rewards, early product access, and dedicated support.",
    },
    1: {
        "name": "Active Core Buyers",
        "color": "#10B981",  # Emerald Green
        "badge_bg": "linear-gradient(135deg, #10B981, #34D399)",
        "description": "Recent purchasers with steady engagement and healthy order values.",
        "strategy": "Drive cross-selling, volume discounts, and personalized recommendations.",
    },
    2: {
        "name": "At-Risk / Dormant",
        "color": "#EF4444",  # Red
        "badge_bg": "linear-gradient(135deg, #EF4444, #F87171)",
        "description": "High recency (haven't bought in a long time) and lower purchase frequency.",
        "strategy": "Launch win-back re-engagement email campaigns with time-limited discounts.",
    },
    3: {
        "name": "Wholesale / Institutional Champions",
        "color": "#F59E0B",  # Gold
        "badge_bg": "linear-gradient(135deg, #F59E0B, #FBBF24)",
        "description": "Ultra-high average order value (AOV) and monumental order totals.",
        "strategy": "Assign dedicated B2B account manager and offer custom bulk contract pricing.",
    },
}

# --- Sidebar ---
st.sidebar.markdown("## 🧬")
st.sidebar.title("Evolutionary AI")
st.sidebar.markdown("**Customer Segmentation Engine**")
st.sidebar.markdown("---")

df_raw = load_data(DATA_PATH)

if df_raw is None:
    st.error(f"Dataset not found at `{DATA_PATH}`.")
    st.info("Please run `src/data_loader.py` and `src/genetic_algorithm.py` to generate the segmented dataset.")
    st.stop()

# Sidebar Filters
st.sidebar.subheader("Filter Segments")
available_clusters = sorted(df_raw["Cluster"].unique().tolist())
selected_clusters = st.sidebar.multiselect(
    "Select Clusters to Display:",
    options=available_clusters,
    default=available_clusters,
    format_func=lambda c: f"Cluster {c} - {CLUSTER_PERSONAS.get(c, {}).get('name', 'Segment ' + str(c))}",
)

filtered_df = df_raw[df_raw["Cluster"].isin(selected_clusters)]

st.sidebar.markdown("---")
st.sidebar.subheader("GA Optimization Spec")
st.sidebar.markdown(
    r"""
- **Algorithm**: DEAP Genetic Algorithm (`eaMuPlusLambda`)
- **Objectives**: Maximize Silhouette Score & Minimize WCSS
- **Feature Space**: Standardized RFM + AOV
- **Dimensionality**: PCA ($\ge 85\%$ Variance)
"""
)

# --- Header ---
st.markdown('<div class="main-title">🧬 Evolutionary Customer Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-driven customer clustering optimizing Silhouette Score & WCSS using Genetic Algorithms</div>',
    unsafe_allow_html=True,
)

# --- Executive KPI Cards ---
total_revenue = filtered_df["Monetary"].sum()
active_customers = len(filtered_df)
avg_aov = filtered_df["Monetary"].sum() / filtered_df["Frequency"].sum() if filtered_df["Frequency"].sum() > 0 else 0
avg_recency = filtered_df["Recency"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">£{total_revenue:,.2f}</div>
            <div class="kpi-sub">Across {active_customers:,} selected customers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Customers</div>
            <div class="kpi-value">{active_customers:,}</div>
            <div class="kpi-sub">{(active_customers / len(df_raw)) * 100:.1f}% of total customer base</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Weighted Average Order Value</div>
            <div class="kpi-value">£{avg_aov:,.2f}</div>
            <div class="kpi-sub">Revenue per invoice transaction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Recency</div>
            <div class="kpi-value">{avg_recency:.1f} days</div>
            <div class="kpi-sub">Average days since last purchase</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- PCA Computation ---
pca_df, exp_var = compute_pca(df_raw)
pca_filtered = pca_df[pca_df["Cluster"].astype(int).isin(selected_clusters)]

# Color map for Plotly
color_map = {
    str(c): CLUSTER_PERSONAS.get(c, {}).get("color", "#3B82F6") for c in available_clusters
}

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Interactive PCA Clusters",
        "🎯 Cluster Summaries & Insights",
        "📈 RFM Metric Distributions",
        "📥 Data Export",
    ]
)

# --- TAB 1: Interactive PCA Plots ---
with tab1:
    st.subheader("Principal Component Analysis (PCA) Cluster Visualization")
    st.markdown(
        f"**Cumulative Explained Variance:** `{(exp_var.sum() * 100):.2f}%` "
        f"(PC1: `{exp_var[0]*100:.1f}%`, PC2: `{exp_var[1]*100:.1f}%`, PC3: `{exp_var[2]*100:.1f}%`)"
    )

    plot_type = st.radio(
        "Plot Dimension:",
        ["3D PCA Space Scatter", "2D Biplot View (PC1 vs PC2)"],
        horizontal=True,
    )

    if plot_type == "3D PCA Space Scatter":
        fig_3d = px.scatter_3d(
            pca_filtered,
            x="PC1",
            y="PC2",
            z="PC3",
            color="Cluster",
            color_discrete_map=color_map,
            hover_data=["Customer ID", "Recency", "Frequency", "Monetary", "AOV"],
            opacity=0.8,
            size_max=10,
            title="3D GA-Optimized Centroid Cluster Distribution",
        )
        fig_3d.update_layout(
            height=650,
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(title="Cluster ID"),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig_2d = px.scatter(
            pca_filtered,
            x="PC1",
            y="PC2",
            color="Cluster",
            color_discrete_map=color_map,
            hover_data=["Customer ID", "Recency", "Frequency", "Monetary", "AOV"],
            opacity=0.75,
            title="2D PCA Projection (PC1 vs PC2)",
        )
        fig_2d.update_layout(
            height=550,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_2d, use_container_width=True)

# --- TAB 2: Cluster Summaries & Insights ---
with tab2:
    st.subheader("Marketing Persona Profiles & Recommendations")

    # Aggregate summary stats
    summary_df = (
        filtered_df.groupby("Cluster")
        .agg(
            Customer_Count=("Customer ID", "count"),
            Mean_Recency_Days=("Recency", "mean"),
            Mean_Frequency_Orders=("Frequency", "mean"),
            Total_Revenue=("Monetary", "sum"),
            Mean_Monetary=("Monetary", "mean"),
            Mean_AOV=("AOV", "mean"),
        )
        .reset_index()
    )

    summary_df["Customer_Share_%"] = (
        (summary_df["Customer_Count"] / len(filtered_df)) * 100
    ).round(2)

    # Format numeric columns for display
    display_summary = summary_df.copy()
    display_summary["Mean_Recency_Days"] = display_summary["Mean_Recency_Days"].round(1)
    display_summary["Mean_Frequency_Orders"] = display_summary["Mean_Frequency_Orders"].round(1)
    display_summary["Total_Revenue"] = display_summary["Total_Revenue"].map("£{:,.2f}".format)
    display_summary["Mean_Monetary"] = display_summary["Mean_Monetary"].map("£{:,.2f}".format)
    display_summary["Mean_AOV"] = display_summary["Mean_AOV"].map("£{:,.2f}".format)

    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Segment Action Cards")

    cols = st.columns(len(selected_clusters)) if selected_clusters else [st]

    for idx, cluster_id in enumerate(sorted(selected_clusters)):
        persona = CLUSTER_PERSONAS.get(
            cluster_id,
            {
                "name": f"Segment {cluster_id}",
                "color": "#3B82F6",
                "badge_bg": "#3B82F6",
                "description": "N/A",
                "strategy": "N/A",
            },
        )
        stats = summary_df[summary_df["Cluster"] == cluster_id]

        if not stats.empty:
            count = stats["Customer_Count"].values[0]
            rev = stats["Total_Revenue"].values[0]
            rec = stats["Mean_Recency_Days"].values[0]
            freq = stats["Mean_Frequency_Orders"].values[0]
            aov = stats["Mean_AOV"].values[0]
        else:
            count, rev, rec, freq, aov = 0, 0, 0, 0, 0

        with cols[idx % len(cols)]:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 16px; height: 100%;">
                    <div class="persona-badge" style="background: {persona['badge_bg']};">
                        Cluster {cluster_id}: {persona['name']}
                    </div>
                    <p style="color: #9CA3AF; font-size: 0.85rem; margin-top: 8px;">{persona['description']}</p>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                    <p style="font-size: 0.82rem; margin: 4px 0;"><b>Customers:</b> {count:,}</p>
                    <p style="font-size: 0.82rem; margin: 4px 0;"><b>Total Spend:</b> £{rev:,.2f}</p>
                    <p style="font-size: 0.82rem; margin: 4px 0;"><b>Avg Recency:</b> {rec:.1f} days</p>
                    <p style="font-size: 0.82rem; margin: 4px 0;"><b>Avg Frequency:</b> {freq:.1f} orders</p>
                    <p style="font-size: 0.82rem; margin: 4px 0;"><b>Avg AOV:</b> £{aov:,.2f}</p>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 12px 0;">
                    <p style="font-size: 0.82rem; color: #6EE7B7;"><b>💡 Strategy:</b> {persona['strategy']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- TAB 3: RFM Metric Distributions ---
with tab3:
    st.subheader("RFM & AOV Feature Distributions Across Clusters")

    metric_choice = st.selectbox(
        "Select Metric to Analyze:",
        options=["Recency", "Frequency", "Monetary", "AOV"],
        index=0,
    )

    use_log = st.checkbox("Apply Log Scale (Recommended for Monetary & Frequency)", value=True if metric_choice in ["Monetary", "Frequency", "AOV"] else False)

    fig_box = px.box(
        filtered_df,
        x="Cluster",
        y=metric_choice,
        color="Cluster",
        color_discrete_map=color_map,
        points="outliers",
        log_y=use_log,
        title=f"{metric_choice} Box Plot by Cluster",
    )
    fig_box.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 4: Data Export ---
with tab4:
    st.subheader("Download Segmented Datasets & Summaries")

    st.markdown("Download full customer cluster assignments or aggregated cluster statistics for marketing integration.")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("### 📄 Aggregated Cluster Summary")
        summary_export_df = (
            filtered_df.groupby("Cluster")
            .agg(
                Customer_Count=("Customer ID", "count"),
                Mean_Recency_Days=("Recency", "mean"),
                Mean_Frequency_Orders=("Frequency", "mean"),
                Total_Revenue=("Monetary", "sum"),
                Mean_Monetary=("Monetary", "mean"),
                Mean_AOV=("AOV", "mean"),
            )
            .reset_index()
        )

        csv_summary_buf = io.BytesIO()
        summary_export_df.to_csv(csv_summary_buf, index=False)
        csv_summary_buf.seek(0)

        st.download_button(
            label="📥 Download Cluster Summary CSV",
            data=csv_summary_buf,
            file_name="cluster_summary_metrics.csv",
            mime="text/csv",
        )

    with col_exp2:
        st.markdown("### 👥 Full Segmented Customer Database")
        csv_full_buf = io.BytesIO()
        filtered_df.to_csv(csv_full_buf, index=False)
        csv_full_buf.seek(0)

        st.download_button(
            label="📥 Download Segmented Customers CSV",
            data=csv_full_buf,
            file_name="segmented_customers_export.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Interactive Customer Explorer")
    search_id = st.text_input("Search by Customer ID:")
    if search_id:
        try:
            cid = int(search_id)
            searched_df = filtered_df[filtered_df["Customer ID"] == cid]
            st.dataframe(searched_df, use_container_width=True)
        except ValueError:
            st.warning("Please enter a valid numeric Customer ID.")
    else:
        st.dataframe(filtered_df.head(100), use_container_width=True)
