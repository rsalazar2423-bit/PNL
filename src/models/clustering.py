"""
============================================================
Módulo: models/clustering.py
============================================================
Agrupamiento semántico no supervisado mediante K-Means.

Responsabilidad Única:
    Ejecutar el modelado K-Means (K óptimo, ajuste y asignación).
    Delega visualización (t-SNE) y validación (métricas) a otros módulos.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.models.projection import project_embeddings_tsne, get_stratified_sample_indices
from src.models.metrics import evaluate_clustering_metrics


def _get_optimal_k(embeddings: np.ndarray, k_range: range = range(3, 9)) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Determina el número óptimo de clusters usando Silhouette Score.
    """
    from sklearn.metrics import silhouette_score
    
    print(f"   [CLUSTERING] Buscando K óptimo en rango {list(k_range)}...")
    best_k = 3
    max_score = -1
    silhouette_scores = []
    
    sample_size = min(2000, embeddings.shape[0])
    indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings[indices], labels[indices])
        
        print(f"      - K={k} | Silhouette: {score:.4f}")
        silhouette_scores.append((k, score))
        if score > max_score:
            max_score = score
            best_k = k
            
    return best_k, silhouette_scores


def _fit_kmeans(embeddings: np.ndarray, k: int) -> KMeans:
    """
    Entrena el modelo KMeans con el número de clústeres indicado.
    """
    print(f"   [CLUSTERING] Ejecutando K-Means con K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    return kmeans


def _get_cluster_labels(df: pd.DataFrame, kmeans: KMeans, embeddings: np.ndarray) -> Dict[int, str]:
    """
    Genera etiquetas descriptivas basadas en el comentario representativo (centroide).
    """
    labels = {}
    for i, centroid in enumerate(kmeans.cluster_centers_):
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        representative_text = df.iloc[closest_idx]['text_clean'][:50] + "..."
        labels[i] = f"Tema {i+1}: {representative_text}"
    return labels


def _run_final_evaluation(
    df: pd.DataFrame, 
    embeddings: np.ndarray, 
    kmeans: KMeans, 
    best_k: int, 
    sil_scores: List[Tuple[int, float]]
) -> Dict[str, Any]:
    """
    Orquesta el cálculo de proyecciones visuales t-SNE y métricas estadísticas.
    """
    # 1. Obtener etiquetas descriptivas y muestras
    cluster_labels = _get_cluster_labels(df, kmeans, embeddings)
    sample_indices = get_stratified_sample_indices(df, best_k)
    coords_3d = project_embeddings_tsne(embeddings, sample_indices)

    # 2. Calcular métricas finales
    print("   [CLUSTERING] Calculando métricas de evaluación final...")
    n_total = len(df)
    eval_sample_size = min(2000, n_total)
    eval_indices = np.random.RandomState(42).choice(n_total, eval_sample_size, replace=False)
    metrics = evaluate_clustering_metrics(embeddings[eval_indices], df.loc[eval_indices, 'cluster'].values)

    print(f"      - Final Silhouette Score: {metrics['silhouette']:.4f}")
    print(f"      - Final Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
    print(f"      - Final Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f}")

    return {
        'best_k': best_k,
        'cluster_labels': cluster_labels,
        'tsne_3d_coords': coords_3d,
        'tsne_sample_idx': sample_indices,
        'silhouette_scores': sil_scores,
        'metrics': metrics
    }


def compute_clusters(df: pd.DataFrame, embeddings: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Orquestador del pipeline de clustering semántico.
    """
    print("\n" + "=" * 60)
    print("🧩 PASO 6: Agrupamiento Semántico (Clustering)")
    print("=" * 60)

    # 1. Encontrar K óptimo (Delegado)
    best_k, sil_scores = _get_optimal_k(embeddings)

    # 2. Ejecutar K-Means final (Delegado)
    kmeans = _fit_kmeans(embeddings, best_k)
    df['cluster'] = kmeans.labels_

    # 3. Generar proyecciones y validaciones (Delegado)
    results = _run_final_evaluation(df, embeddings, kmeans, best_k, sil_scores)

    print("✅ [CLUSTERING] Segmentación semántica finalizada.")
    return df, results
