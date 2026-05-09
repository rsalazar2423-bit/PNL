"""
============================================================
Módulo: models/clustering.py
============================================================
Agrupamiento no supervisado por temas latentes.

Responsabilidad Única:
    Ejecutar K-Means, LDA y t-SNE para descubrir agrupaciones
    temáticas naturales en el corpus. Retorna datos numéricos
    puros (coordenadas, etiquetas, keywords). NO genera gráficos.

Ejemplo de uso:
    >>> from src.models.clustering import compute_clusters
    >>> df, result = compute_clusters(df, tfidf_matrix, feature_names)
    >>> print(f"Clusters óptimos: {result['best_k']}")
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def _find_optimal_k(tfidf_matrix, k_range=range(3, 9)) -> tuple:
    """
    Encuentra el número óptimo de clusters maximizando el Silhouette Score.

    Args:
        tfidf_matrix: Matriz dispersa TF-IDF.
        k_range (range): Rango de Ks a evaluar.

    Returns:
        tuple: (Mejor K encontrado, Lista de (K, score)).
    """
    print(f"   [CLUSTERING] Evaluando el K óptimo en el rango {list(k_range)}...")
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(
            tfidf_matrix, labels,
            sample_size=min(2000, tfidf_matrix.shape[0])
        )
        scores.append((k, score))
        print(f"      - K={k} -> Silhouette: {score:.4f}")

    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"   [CLUSTERING] K óptimo seleccionado: {best_k}")
    return best_k, scores


def _run_lda(df: pd.DataFrame, n_topics: int = 5) -> list:
    """
    Ejecuta Latent Dirichlet Allocation para descubrir temas latentes.

    Args:
        df (pd.DataFrame): Dataset con la columna 'lemmas_text'.
        n_topics (int): Número de temas a extraer. Default: 5.

    Returns:
        list[dict]: Lista de temas, cada uno con:
            - 'topic_id' (int): Índice del tema.
            - 'words' (list[tuple]): Top 10 palabras con su peso.
            - 'label' (str): Etiqueta legible (Top 3 palabras).
    """
    print(f"   [CLUSTERING] Ejecutando LDA para {n_topics} temas...")
    count_vec = CountVectorizer(max_features=3000, max_df=0.85, min_df=3)
    count_matrix = count_vec.fit_transform(df['lemmas_text'])
    names = count_vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=20, learning_method='online', batch_size=256
    )
    lda.fit(count_matrix)

    topics = []
    for tid, dist in enumerate(lda.components_):
        norm = dist / dist.sum()
        top_idx = norm.argsort()[-10:][::-1]
        words = [(names[i], float(norm[i])) for i in top_idx]
        label = " | ".join([w for w, _ in words[:3]])
        topics.append({'topic_id': tid, 'words': words, 'label': f"Tema {tid + 1}: {label}"})

    print("   [CLUSTERING] LDA finalizado.")
    return topics


def compute_clusters(df: pd.DataFrame, tfidf_matrix, feature_names) -> tuple:
    """
    Ejecuta el flujo completo de clustering:
    1. K-Óptimo (Silhouette)
    2. K-Means
    3. LDA
    4. Reducción t-SNE 3D

    Args:
        df (pd.DataFrame): Dataset base.
        tfidf_matrix: Matriz TF-IDF.
        feature_names: Nombres de las características del TF-IDF.

    Returns:
        tuple: (DataFrame con columna 'cluster', dict con claves:
            - 'best_k' (int): Número óptimo de clusters.
            - 'silhouette_scores' (list[tuple]): Puntuaciones por K.
            - 'cluster_keywords' (dict): {cluster_id: top_10_palabras}.
            - 'lda_topics' (list[dict]): Temas latentes descubiertos.
            - 'tsne_3d_coords' (ndarray): Coordenadas t-SNE 3D.
            - 'tsne_sample_idx' (list[int]): Índices de las muestras en t-SNE.
        )
    """
    print("\n" + "=" * 60)
    print("🧩 PASO 6: Agrupamiento No Supervisado (Clustering)")
    print("=" * 60)

    best_k, sil_scores = _find_optimal_k(tfidf_matrix)

    print(f"   [CLUSTERING] Ejecutando K-Means con K={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)

    print("   [CLUSTERING] Extrayendo keywords por clúster...")
    cluster_keywords = {}
    feature_names_list = list(feature_names)
    for c in range(best_k):
        center = kmeans.cluster_centers_[c]
        top_idx = center.argsort()[-10:][::-1]
        cluster_keywords[c] = [(feature_names_list[i], center[i]) for i in top_idx]

    lda_topics = _run_lda(df, n_topics=best_k)

    # t-SNE 3D con muestreo estratificado
    print("   [CLUSTERING] Calculando t-SNE 3D...")
    n_samples = min(500, tfidf_matrix.shape[0])
    sample_idx = []
    for c in range(best_k):
        cluster_idx = df[df['cluster'] == c].index.tolist()
        n_take = min(len(cluster_idx), n_samples // best_k)
        sample_idx.extend(np.random.choice(cluster_idx, n_take, replace=False).tolist())
    sample_idx = sorted(sample_idx)

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)
    tsne_coords = tsne.fit_transform(tfidf_matrix[sample_idx].toarray())

    print("✅ [CLUSTERING] Clustering completado.")
    return df, {
        'best_k': best_k,
        'silhouette_scores': sil_scores,
        'cluster_keywords': cluster_keywords,
        'lda_topics': lda_topics,
        'tsne_3d_coords': tsne_coords,
        'tsne_sample_idx': sample_idx,
    }
