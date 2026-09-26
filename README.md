# evolutionary-customer-segmentation
Hybrid Genetic Algorithm + PCA + K-Means pipeline for E-Commerce Customer Segmentation and Persona Profiling.
# Evolutionary AI Customer Segmentation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_LIVE_URL_HERE)

An end-to-end Machine Learning project combining RFM analysis, PCA dimensionality reduction, and a DEAP-based Genetic Algorithm (`eaMuPlusLambda`) to optimize K-Means cluster centroids.

## 📌 Problem Statement
Standard K-Means clustering often gets trapped in local optima due to random centroid initialization. This engine uses an evolutionary search space to maximize Silhouette Score while minimizing Within-Cluster Sum of Squares (WCSS), identifying 4 distinct behavioral segments from e-commerce transaction data.

## 📊 Dataset
- **Source:** [UCI Machine Learning Repository - Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

## 🧬 Methodology
1. **Feature Engineering:** Recency, Frequency, Monetary (RFM), and Average Order Value (AOV).
2. **Dimensionality Reduction:** `StandardScaler` + `PCA` retaining 92% explained variance.
3. **Evolutionary Optimization:** Custom fitness evaluation using `DEAP` (`eaMuPlusLambda`) to evolve centroid coordinates.
4. **Interactive Dashboard:** Built with Streamlit & Plotly for 3D cluster exploration.

## 🎯 Target Customer Segments
- **Cluster 0 — VIP High Spenders:** High monetary value and high order frequency.
- **Cluster 1 — Active Core Buyers:** Steady purchasing cadence with moderate order values.
- **Cluster 2 — At-Risk / Dormant:** High recency (long time since last purchase); targeted win-back campaigns needed.
- **Cluster 3 — Wholesale / Institutional Champions:** High weighted AOV (£479.95) with lower frequency; ideal for dedicated account management.

## 🚀 Local Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py

