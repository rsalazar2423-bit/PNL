"""
============================================================
Módulo: models/projection.py
============================================================
Reducción de dimensionalidad y muestreo representativo.

Responsabilidad Única:
    Proyectar embeddings de alta dimensión a espacios de baja dimensión (t-SNE)
    para visualizaciones, y realizar muestreos estratificados.
"""

from typing import List
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def get_stratified_sample_indices(df: pd.DataFrame, best_k: int, n_sample: int = 800) -> List[int]:
    """
    Obtiene una muestra estratificada de índices representativos de cada clúster.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'cluster'.
        best_k (int): Número de clústeres.
        n_sample (int): Tamaño total de la muestra deseada.

    Returns:
        List[int]: Índices seleccionados de forma reproducible.
    """
    target_per_cluster = max(1, n_sample // best_k)
    sample_indices = []
    for _, group in df.groupby('cluster'):
        group_size = len(group)
        sample_size = min(group_size, target_per_cluster)
        sampled_group = group.sample(n=sample_size, random_state=42)
        sample_indices.extend(sampled_group.index.tolist())
    return sorted(sample_indices)


def project_embeddings_tsne(embeddings: np.ndarray, sample_indices: List[int]) -> np.ndarray:
    """
    Reduce la dimensionalidad de los embeddings a 3D usando t-SNE.

    Args:
        embeddings (np.ndarray): Matriz completa de vectores.
        sample_indices (List[int]): Índices a proyectar.

    Returns:
        np.ndarray: Coordenadas 3D proyectadas.
    """
    print(f"   [PROJECTION] Calculando proyección t-SNE 3D para {len(sample_indices)} puntos...")
    tsne = TSNE(
        n_components=3, 
        perplexity=min(30, len(sample_indices) - 1), 
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    return tsne.fit_transform(embeddings[sample_indices])
