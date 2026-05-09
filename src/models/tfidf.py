"""
============================================================
Módulo: models/tfidf.py
============================================================
Vectorización TF-IDF y extracción de n-gramas del corpus.

Responsabilidad Única:
    Transformar los textos lematizados en una matriz TF-IDF y
    extraer los términos, bigramas y trigramas más relevantes.
    NO genera gráficos — solo retorna datos numéricos.

Ejemplo de uso:
    >>> from src.models.tfidf import compute_tfidf
    >>> result = compute_tfidf(df)
    >>> print(result['top_terms'][:5])
"""

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def compute_tfidf(df: pd.DataFrame, max_features: int = 5000, top_n: int = 30) -> dict:
    """
    Ejecuta la vectorización TF-IDF sobre los textos lematizados.

    Args:
        df (pd.DataFrame): DataFrame con la columna 'lemmas_text'.
        max_features (int): Límite máximo de características. Default: 5000.
        top_n (int): Cantidad de términos top a retornar. Default: 30.

    Returns:
        dict: Diccionario con claves:
            - 'tfidf_matrix' (sparse matrix): Matriz TF-IDF (n_docs × n_features).
            - 'vectorizer' (TfidfVectorizer): Vectorizador entrenado.
            - 'feature_names' (ndarray): Nombres de las características.
            - 'top_terms' (list[tuple]): Top términos como (término, score_promedio).
            - 'bigrams' (list[tuple]): Top 15 bigramas como (bigrama, frecuencia).
            - 'trigrams' (list[tuple]): Top 10 trigramas como (trigrama, frecuencia).
            - 'pca_3d' (ndarray): Coordenadas PCA 3D de los top términos.
            - 'pca_labels' (list[str]): Nombres de los términos para PCA.
    """
    print("\n" + "=" * 60)
    print("📈 PASO 3: Extracción de Relevancia Semántica (TF-IDF)")
    print("=" * 60)

    print(f"   [TF-IDF] Inicializando TfidfVectorizer (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=3
    )

    tfidf_matrix = vectorizer.fit_transform(df['lemmas_text'])
    feature_names = vectorizer.get_feature_names_out()
    print(f"   [TF-IDF] Matriz generada: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

    # Promedios de impacto global por término
    print(f"   [TF-IDF] Calculando scores promedio de los top {top_n} términos...")
    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], mean_scores[i]) for i in top_indices]

    # Bigramas y trigramas
    print("   [TF-IDF] Extrayendo bigramas y trigramas...")
    all_lemmas_text = ' '.join(df['lemmas_text'].tolist())
    words = all_lemmas_text.split()

    bigrams = Counter(zip(words[:-1], words[1:])).most_common(15)
    bigrams = [(' '.join(bg), count) for bg, count in bigrams]

    trigrams = Counter(zip(words[:-2], words[1:-1], words[2:])).most_common(10)
    trigrams = [(' '.join(tg), count) for tg, count in trigrams]

    # PCA 3D de los vectores TF-IDF de los top términos
    print("   [TF-IDF] Calculando reducción PCA 3D...")
    top_feature_indices = mean_scores.argsort()[-50:][::-1]
    top_vectors = tfidf_matrix[:, top_feature_indices].toarray().T

    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(top_vectors)
    pca_labels = [feature_names[i] for i in top_feature_indices]

    print("✅ [TF-IDF] Análisis TF-IDF completado.")
    return {
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'feature_names': feature_names,
        'top_terms': top_terms,
        'bigrams': bigrams,
        'trigrams': trigrams,
        'pca_3d': pca_3d,
        'pca_labels': pca_labels,
    }
