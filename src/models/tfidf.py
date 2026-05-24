"""
============================================================
Módulo: models/tfidf.py
============================================================
Vectorización TF-IDF y extracción de n-gramas del corpus.

Responsabilidad Única:
    Transformar los textos lematizados en una matriz TF-IDF y
    extraer los términos, bigramas y trigramas más relevantes.
    Modularizado en funciones con responsabilidad única.
"""

from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


def _fit_tfidf(texts: pd.Series, max_features: int) -> tuple:
    """
    Vectoriza los textos usando TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        max_df=0.85,
        min_df=3
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, vectorizer, feature_names


def _get_top_terms(tfidf_matrix, feature_names, top_n: int) -> tuple:
    """
    Calcula los términos con mayor score TF-IDF promedio global.
    """
    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], mean_scores[i]) for i in top_indices]
    return top_terms, mean_scores


def _extract_ngrams(df: pd.DataFrame, top_bigrams: int = 15, top_trigrams: int = 10) -> tuple:
    """
    Extrae bigramas y trigramas más comunes calculados por comentario.
    """
    bigram_counter = Counter()
    trigram_counter = Counter()
    
    for lemmas_text in df['lemmas_text'].dropna():
        words = lemmas_text.split()
        if len(words) >= 2:
            bigram_counter.update(zip(words[:-1], words[1:]))
        if len(words) >= 3:
            trigram_counter.update(zip(words[:-2], words[1:-1], words[2:]))
            
    bigrams = [(' '.join(bg), count) for bg, count in bigram_counter.most_common(top_bigrams)]
    trigrams = [(' '.join(tg), count) for tg, count in trigram_counter.most_common(top_trigrams)]
    return bigrams, trigrams


def _compute_pca_3d(tfidf_matrix, feature_names, mean_scores, n_top_terms: int = 50) -> tuple:
    """
    Calcula la reducción de dimensionalidad PCA 3D de los términos más importantes.
    """
    top_feature_indices = mean_scores.argsort()[-n_top_terms:][::-1]
    top_vectors = tfidf_matrix[:, top_feature_indices].toarray().T
    
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(top_vectors)
    pca_labels = [feature_names[i] for i in top_feature_indices]
    return pca_3d, pca_labels


def compute_tfidf(df: pd.DataFrame, max_features: int = 5000, top_n: int = 30) -> dict:
    """
    Función orquestadora del análisis TF-IDF.
    """
    print("\n" + "=" * 60)
    print("📈 PASO 3: Extracción de Relevancia Semántica (TF-IDF)")
    print("=" * 60)

    print(f"   [TF-IDF] Ajustando matriz TF-IDF...")
    tfidf_matrix, vectorizer, feature_names = _fit_tfidf(df['lemmas_text'], max_features)
    print(f"   [TF-IDF] Matriz generada: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")

    print(f"   [TF-IDF] Calculando relevancia promedio...")
    top_terms, mean_scores = _get_top_terms(tfidf_matrix, feature_names, top_n)

    print("   [TF-IDF] Extrayendo bigramas y trigramas...")
    bigrams, trigrams = _extract_ngrams(df)

    print("   [TF-IDF] Calculando reducción PCA 3D...")
    pca_3d, pca_labels = _compute_pca_3d(tfidf_matrix, feature_names, mean_scores)

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
