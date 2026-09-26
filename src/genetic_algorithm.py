import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from deap import base, creator, tools, algorithms

# Define Project Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_RFM_PATH = os.path.join(DATA_DIR, "processed_rfm.csv")
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "segmented_customers.csv")


def load_rfm_data(file_path: str = DEFAULT_RFM_PATH) -> pd.DataFrame:
    """
    Loads processed RFM dataset from CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Processed RFM dataset not found at '{file_path}'. "
            f"Please run data_loader.py first."
        )

    df = pd.read_csv(file_path)
    print(f"Loaded RFM dataset with shape: {df.shape}")
    return df


def preprocess_and_pca(
    df: pd.DataFrame,
    feature_cols: list = None,
    variance_threshold: float = 0.85,
    random_state: int = 42,
):
    """
    Standardizes feature columns and applies PCA to retain >= variance_threshold (85%) variance.

    Returns:
    - X_pca: Transformed features in PCA space
    - scaler: Fitted StandardScaler object
    - pca: Fitted PCA object
    """
    if feature_cols is None:
        feature_cols = ["Recency", "Frequency", "Monetary", "AOV"]

    X = df[feature_cols].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA retaining >= variance_threshold
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_.sum()
    print(
        f"PCA reduced dimensions from {X.shape[1]} to {X_pca.shape[1]} "
        f"retaining {explained_var * 100:.2f}% cumulative variance."
    )

    return X_pca, scaler, pca


def evaluate_centroids(individual, X_pca: np.ndarray, k: int, sample_size: int = 1000):
    """
    Evaluates a candidate solution (chromosome of k cluster centroids).

    Returns multi-objective fitness:
    - Silhouette Score (Maximize, weight: +1.0)
    - WCSS / Inertia (Minimize, weight: -1.0)
    """
    d = X_pca.shape[1]
    centroids = np.array(individual).reshape(k, d)

    # Compute Euclidean distances from all points to centroids: shape (N, k)
    dists = np.linalg.norm(X_pca[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    labels = np.argmin(dists, axis=1)

    # Check for empty / degenerate clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < k:
        # Penalize degenerate solutions that lose clusters
        return -1.0, 1e9

    # Calculate WCSS (Within-Cluster Sum of Squares)
    wcss = float(np.sum(np.min(dists, axis=1) ** 2))

    # Calculate Silhouette Score on a random sample to maintain fast evaluation
    if sample_size and len(X_pca) > sample_size:
        sample_indices = np.random.choice(len(X_pca), size=sample_size, replace=False)
        sil_score = float(silhouette_score(X_pca[sample_indices], labels[sample_indices]))
    else:
        sil_score = float(silhouette_score(X_pca, labels))

    return sil_score, wcss


def initialize_centroids_individual(X_pca: np.ndarray, k: int):
    """
    Initializes an individual by randomly sampling k data points from X_pca as initial centroids.
    """
    d = X_pca.shape[1]
    indices = np.random.choice(len(X_pca), size=k, replace=False)
    centroids = X_pca[indices].flatten()
    return centroids.tolist()


def run_genetic_algorithm(
    X_pca: np.ndarray,
    k: int = 4,
    pop_size: int = 50,
    n_gen: int = 30,
    cx_pb: float = 0.7,
    mut_pb: float = 0.2,
    sample_size: int = 1000,
    seed: int = 42,
):
    """
    Runs a Genetic Algorithm using DEAP to optimize K-Means cluster centroids.

    Optimization objectives:
    1. Maximize Silhouette Score (+1.0)
    2. Minimize WCSS (-1.0)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Define DEAP fitness and individual types
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # Register individual generator
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        lambda: initialize_centroids_individual(X_pca, k),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register(
        "evaluate", evaluate_centroids, X_pca=X_pca, k=k, sample_size=sample_size
    )

    # Register genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    # Initialize Population
    pop = toolbox.population(n=pop_size)

    # Hall of Fame / Pareto Front
    pareto_front = tools.ParetoFront()

    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)

    print(f"Starting Genetic Algorithm optimization (k={k}, pop_size={pop_size}, n_gen={n_gen})...")
    pop, logbook = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cx_pb,
        mutpb=mut_pb,
        ngen=n_gen,
        stats=stats,
        halloffame=pareto_front,
        verbose=True,
    )

    # Find the individual with the highest Silhouette Score from Pareto Front / Population
    best_ind = None
    best_sil = -1.0

    for ind in pareto_front:
        sil, wcss = ind.fitness.values
        if sil > best_sil:
            best_sil = sil
            best_ind = ind

    if best_ind is None:
        best_ind = tools.selBest(pop, 1)[0]

    # Calculate full-dataset metrics for best solution
    d = X_pca.shape[1]
    best_centroids = np.array(best_ind).reshape(k, d)
    dists = np.linalg.norm(X_pca[:, np.newaxis, :] - best_centroids[np.newaxis, :, :], axis=2)
    final_labels = np.argmin(dists, axis=1)
    full_wcss = float(np.sum(np.min(dists, axis=1) ** 2))
    full_sil = float(silhouette_score(X_pca, final_labels))

    print(f"\nGA Optimization Completed!")
    print(f"Best Solution Full Silhouette Score: {full_sil:.4f}")
    print(f"Best Solution Full WCSS: {full_wcss:.2f}")

    return best_centroids, final_labels, full_sil, full_wcss


def segment_customers(
    rfm_path: str = DEFAULT_RFM_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    k: int = 4,
    pop_size: int = 50,
    n_gen: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Full workflow execution:
    1. Load data/processed_rfm.csv
    2. Standardize features and apply PCA (retaining >= 85% variance)
    3. Run GA using DEAP to optimize cluster centroids
    4. Save cluster assignments to data/segmented_customers.csv
    """
    df_rfm = load_rfm_data(rfm_path)
    X_pca, scaler, pca = preprocess_and_pca(df_rfm, variance_threshold=0.85, random_state=seed)

    best_centroids, labels, sil_score, wcss = run_genetic_algorithm(
        X_pca, k=k, pop_size=pop_size, n_gen=n_gen, seed=seed
    )

    df_segmented = df_rfm.copy()
    df_segmented["Cluster"] = labels

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df_segmented.to_csv(output_path, index=False)
    print(f"Segmented customer dataset saved successfully to: {output_path}")

    return df_segmented


if __name__ == "__main__":
    segment_customers()
