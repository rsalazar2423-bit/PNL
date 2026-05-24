"""
============================================================
Módulo: models/metrics.py
============================================================
Métricas de evaluación de modelos y calidad de agrupamientos.

Responsabilidad Única:
    Calcular métricas de validación estadística (Silhouette, Davies-Bouldin, etc.).
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evalúa la calidad del agrupamiento semántico.

    Args:
        embeddings (np.ndarray): Matriz de vectores.
        labels (np.ndarray): Etiquetas asignadas a cada vector.

    Returns:
        dict: Silueta (Silhouette), Davies-Bouldin (DBI) y Calinski-Harabasz (CHI).
    """
    if len(np.unique(labels)) > 1:
        sil = float(silhouette_score(embeddings, labels))
        dbi = float(davies_bouldin_score(embeddings, labels))
        chi = float(calinski_harabasz_score(embeddings, labels))
    else:
        sil, dbi, chi = 0.0, 0.0, 0.0

    return {
        'silhouette': sil,
        'davies_bouldin': dbi,
        'calinski_harabasz': chi
    }


def calculate_vocab_compression(df: pd.DataFrame) -> float:
    """
    Calcula el porcentaje de compresión de vocabulario tras la limpieza.
    """
    try:
        raw_words = set()
        for t in df['text'].dropna():
            raw_words.update(re.findall(r'\b\w+\b', str(t).lower()))
        raw_vocab_size = len(raw_words)
        
        clean_words = set()
        for lemmas_list in df['lemmas'].dropna():
            clean_words.update(lemmas_list)
        clean_vocab_size = len(clean_words)
        return (1.0 - (clean_vocab_size / raw_vocab_size)) * 100 if raw_vocab_size > 0 else 0.0
    except Exception:
        return 0.0


def calculate_missing_clustering_metrics_if_needed(df: pd.DataFrame, cluster_results: dict) -> dict:
    """
    Calcula dinámicamente las métricas de clustering si no están presentes en la caché.
    """
    cluster_metrics = cluster_results.get('metrics', {})
    if (not cluster_metrics or cluster_metrics.get('silhouette', 0.0) == 0.0) and 'cluster' in df.columns:
        print("   [METRICS] Métricas de clustering faltantes en caché. Calculando dinámicamente sobre una muestra...")
        try:
            from src.models.embeddings import compute_embeddings
            from sklearn.cluster import KMeans
            
            n_total = len(df)
            eval_sample_size = min(1000, n_total)
            np.random.seed(42)
            eval_indices = np.random.choice(n_total, eval_sample_size, replace=False)
            sample_texts = df.loc[eval_indices, 'text_clean'].tolist()
            
            eval_embeddings = compute_embeddings(sample_texts)
            eval_labels = df.loc[eval_indices, 'cluster'].values
            
            cluster_results['metrics'] = evaluate_clustering_metrics(eval_embeddings, eval_labels)
            
            if not cluster_results.get('silhouette_scores'):
                print("   [METRICS] Calculando curva de Silhouette dinámicamente...")
                sil_scores = []
                for k in range(3, 9):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                    labels_k = kmeans.fit_predict(eval_embeddings)
                    score = float(silhouette_score(eval_embeddings, labels_k))
                    sil_scores.append((k, score))
                cluster_results['silhouette_scores'] = sil_scores
                    
        except Exception as ex:
            print(f"   [METRICS] Error al calcular métricas dinámicas: {ex}")
    return cluster_results
