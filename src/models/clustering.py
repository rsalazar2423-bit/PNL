"""
============================================================
Módulo: models/clustering.py
============================================================
Agrupamiento semántico no supervisado mediante Embeddings.

Este módulo identifica temas latentes en el corpus agrupando los 
vectores semánticos generados en la Fase 2. Utiliza K-Means para 
la segmentación y t-SNE para la visualización 3D.

Principios aplicados:
    - SRP: Funciones separadas para reducción, búsqueda de K y clustering.
    - Precisión Semántica: Uso de centroides para etiquetado.
    - Documentación: Google Style docstrings.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def _get_optimal_k(embeddings: np.ndarray, k_range: range = range(3, 9)) -> int:
    """
    Determina el número óptimo de clusters usando Silhouette Score.

    Args:
        embeddings (np.ndarray): Matriz de vectores semánticos.
        k_range (range): Rango de valores K a evaluar.

    Returns:
        int: El mejor valor de K encontrado.
    """
    print(f"   [CLUSTERING] Buscando K óptimo en rango {list(k_range)}...")
    best_k = 3
    max_score = -1
    
    # Muestreo para velocidad en el cálculo de silueta
    sample_size = min(2000, embeddings.shape[0])
    indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
    sample_data = embeddings[indices]

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings[indices], labels[indices])
        
        print(f"      - K={k} | Silhouette: {score:.4f}")
        if score > max_score:
            max_score = score
            best_k = k
            
    return best_k

def _apply_tsne_3d(embeddings: np.ndarray, sample_indices: List[int]) -> np.ndarray:
    """
    Reduce la dimensionalidad de los embeddings a 3D para visualización.

    Args:
        embeddings (np.ndarray): Matriz completa de vectores.
        sample_indices (List[int]): Índices de la muestra a reducir.

    Returns:
        np.ndarray: Coordenadas (N_sample, 3).
    """
    print(f"   [CLUSTERING] Calculando proyección t-SNE 3D para {len(sample_indices)} puntos...")
    tsne = TSNE(
        n_components=3, 
        perplexity=min(30, len(sample_indices) - 1), 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    return tsne.fit_transform(embeddings[sample_indices])

def _get_cluster_labels(df: pd.DataFrame, kmeans: KMeans, embeddings: np.ndarray) -> Dict[int, str]:
    """
    Genera etiquetas descriptivas para cada clúster basándose en el 
    comentario más representativo (más cercano al centroide).

    Args:
        df (pd.DataFrame): Dataset con los textos originales.
        kmeans (KMeans): Modelo entrenado.
        embeddings (np.ndarray): Vectores utilizados.

    Returns:
        Dict: Mapa de {cluster_id: etiqueta_descriptiva}.
    """
    labels = {}
    for i, centroid in enumerate(kmeans.cluster_centers_):
        # Encontrar el índice del punto más cercano al centroide
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        closest_idx = np.argmin(distances)
        representative_text = df.iloc[closest_idx]['text_clean'][:50] + "..."
        labels[i] = f"Tema {i+1}: {representative_text}"
    return labels

def compute_clusters(df: pd.DataFrame, embeddings: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Orquestador del pipeline de clustering semántico.

    Args:
        df (pd.DataFrame): Dataset preprocesado.
        embeddings (np.ndarray): Matriz de vectores generada en Fase 2.

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame con clúster y metadata para gráficos.
    """
    print("\n" + "=" * 60)
    print("🧩 PASO 6: Agrupamiento Semántico (Clustering)")
    print("=" * 60)

    # 1. Encontrar K óptimo
    best_k = _get_optimal_k(embeddings)

    # 2. Ejecutar K-Means final
    print(f"   [CLUSTERING] Ejecutando K-Means con K={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # 3. Obtener etiquetas descriptivas
    cluster_labels = _get_cluster_labels(df, kmeans, embeddings)

    # 4. Muestreo estratificado para t-SNE (por eficiencia visual)
    n_total = len(df)
    n_sample = min(800, n_total)
    sample_indices = df.groupby('cluster').apply(
        lambda x: x.sample(min(len(x), n_sample // best_k))
    ).index.get_level_values(1).tolist()
    sample_indices = sorted(sample_indices)

    # 5. Reducción dimensional
    coords_3d = _apply_tsne_3d(embeddings, sample_indices)

    print("✅ [CLUSTERING] Segmentación semántica finalizada.")
    
    return df, {
        'best_k': best_k,
        'cluster_labels': cluster_labels,
        'tsne_3d_coords': coords_3d,
        'tsne_sample_idx': sample_indices,
        'silhouette_scores': [] # Opcional: podrías devolver todos los scores si los necesitas
    }
