# evolutionary-customer-segmentation
Hybrid Genetic Algorithm + PCA + K-Means pipeline for E-Commerce Customer Segmentation and Persona Profiling.

# 🧬 Evolutionary AI Customer Segmentation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evolutionary-customer-segmentation.streamlit.app)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end Machine Learning pipeline combining RFM+AOV feature engineering, Principal Component Analysis (PCA), and a Genetic Algorithm (`eaMuPlusLambda` via DEAP) to optimize K-Means cluster centroids for e-commerce customer segmentation.

🚀 **Live Interactive App:** [https://evolutionary-customer-segmentation.streamlit.app](https://evolutionary-customer-segmentation.streamlit.app)

📌 Problem Statement
Standard K-Means clustering relies on random centroid initialization (e.g., K-Means++), which can easily trap models in sub-optimal local minima—especially on high-dimensional transaction data.

This engine replaces standard initialization with an Evolutionary Search Algorithm (eaMuPlusLambda) to evolve cluster centroids across a multi-objective fitness function (maximizing Silhouette Score while minimizing Within-Cluster Sum of Squares / WCSS).

🏗️ Architecture & Workflow

┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│   Online Retail II Data │ ──> │   RFM + AOV Feature     │ ──> │   StandardScaler +      │
│   (UCI Repository)      │     │   Engineering           │     │   PCA (92% Variance)    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
                                                                             │
┌─────────────────────────┐     ┌─────────────────────────┐                  ▼
│ Streamlit 3D Dashboard  │ <── │ 4 Behavioral Clusters   │ <── ┌─────────────────────────┐
│ & Persona Insights      │     │ (VIP, Dormant, etc.)    │     │ DEAP Genetic Algorithm  │
└─────────────────────────┘     └─────────────────────────┘     │ (eaMuPlusLambda)        │
                                                                └─────────────────────────┘

📊 Dataset

Source: UCI Machine Learning Repository - Online Retail II

Scope: 500,000+ non-store online retail transactions.

Preprocessing: Filtered non-positive quantities/prices and missing Customer IDs.

🧬 Methodology & Technical Stack

Language & Frameworks: Python 3.10+, Pandas, NumPy, Scikit-Learn, DEAP, Streamlit, Plotly.

Feature Engineering: Recency (days since last purchase), Frequency (total orders), Monetary (total spend), Average Order Value (AOV).

Dimensionality Reduction: PCA retaining 92% explained variance across 3 principal components.

Optimization Algorithm: DEAP eaMuPlusLambda genetic algorithm with two-point crossover (cxTwoPoint), Gaussian mutation (mutGaussian), and tournament selection (selTournament).   

🎯 Target Customer Segments

Cluster ID,Segment Name,Behavioral Profile,Key Business Action

Cluster 0,VIP High Spenders,"High Monetary spend, frequent orders, very recent activity.","Exclusive loyalty perks, early product access."

Cluster 1,Active Core Buyers,"Steady purchase cadence, moderate order value.",Upsell & cross-sell automated email workflows.

Cluster 2,At-Risk / Dormant,Long time since last purchase (high Recency).,Automated win-back campaigns & discount incentives.

Cluster 3,Wholesale / Champions,High weighted AOV (£479.95) with lower frequency.,Dedicated account manager & bulk volume pricing.

💻 Local Setup & Run

git clone [https://github.com/Pushkr57/evolutionary-customers-segmentation.git](https://github.com/Pushkr57/evolutionary-customers-segmentation.git)
cd evolutionary-customers-segmentation

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

streamlit run app/app.py

👥 Authors:

Pushkar Bhogaonkar (@Pushkr57)
Dhanashri Kate (@Dhanashri08)