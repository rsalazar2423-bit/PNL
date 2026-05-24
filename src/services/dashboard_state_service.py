"""
============================================================
Módulo: services/dashboard_state_service.py
============================================================
Administra el estado y carga de datos del dashboard.

Responsabilidad Única: 
    Proveer datos puros del negocio sin saber nada de la UI (Gradio) o Plotly.
"""

import joblib
import pandas as pd
from src.models.metrics import (
    calculate_vocab_compression,
    calculate_missing_clustering_metrics_if_needed
)


def get_dashboard_state() -> dict:
    """
    Carga y procesa los datos y métricas en formato puro de Python.

    Returns:
        dict: Contiene dataframes, modelos e indicadores analíticos puros.
    """
    # 1. Cargar datos del caché
    df = pd.read_parquet("pipeline_data.parquet")
    data = joblib.load('pipeline_models.pkl')
    
    # 2. Asegurar que las métricas de clustering estén completas
    data['cluster_results'] = calculate_missing_clustering_metrics_if_needed(df, data['cluster_results'])
    
    # 3. Calcular métricas estadísticas
    avg_sent_conf = df['sentiment_score'].mean() * 100 if 'sentiment_score' in df.columns else 0.0
    avg_emot_conf = df['emotion_score'].mean() * 100 if 'emotion_score' in df.columns else 0.0
    vocab_compression = calculate_vocab_compression(df)
    
    cluster_metrics = data['cluster_results'].get('metrics', {})
    silhouette = cluster_metrics.get('silhouette', 0.0)
    davies_bouldin = cluster_metrics.get('davies_bouldin', 0.0)
    calinski_harabasz = cluster_metrics.get('calinski_harabasz', 0.0)
    
    # Métricas de Reputación Corporativa (Net Sentiment Score)
    n_pos = len(df[df['sentiment'] == 'POS']) if 'sentiment' in df.columns else 0
    n_neg = len(df[df['sentiment'] == 'NEG']) if 'sentiment' in df.columns else 0
    n_neu = len(df[df['sentiment'] == 'NEU']) if 'sentiment' in df.columns else 0
    total_sent = n_pos + n_neg + n_neu
    
    if total_sent > 0:
        pos_pct = (n_pos / total_sent) * 100
        neg_pct = (n_neg / total_sent) * 100
        nss = pos_pct - neg_pct
        polarization = pos_pct + neg_pct
    else:
        nss = 0.0
        polarization = 0.0
        
    likes_pos = df[df['sentiment'] == 'POS']['likes'].sum() if 'sentiment' in df.columns and 'likes' in df.columns else 0
    likes_neg = df[df['sentiment'] == 'NEG']['likes'].sum() if 'sentiment' in df.columns and 'likes' in df.columns else 0
    likes_neu = df[df['sentiment'] == 'NEU']['likes'].sum() if 'sentiment' in df.columns and 'likes' in df.columns else 0
    total_likes_sent = likes_pos + likes_neg + likes_neu
    nss_weighted = ((likes_pos - likes_neg) / total_likes_sent * 100) if total_likes_sent > 0 else 0.0
    
    # 4. Extraer estadísticas globales de EDA
    total_comments = len(df)
    unique_authors = df['author'].nunique() if 'author' in df.columns else 0
    total_likes = int(df['likes'].sum()) if 'likes' in df.columns else 0
    avg_length = int(round(df['text'].str.len().mean())) if 'text' in df.columns else 0
    
    # Retornar estructura de datos pura
    return {
        'df': df,
        'tfidf_data': data['tfidf'],
        'ner_pos': data['ner_pos'],
        'cluster_results': data['cluster_results'],
        'rag': data['rag'],
        'metrics': {
            'avg_sent_conf': avg_sent_conf,
            'avg_emot_conf': avg_emot_conf,
            'vocab_compression': vocab_compression,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz,
            'nss': nss,
            'nss_weighted': nss_weighted,
            'polarization': polarization
        },
        'stats': {
            'total_comments': total_comments,
            'unique_authors': unique_authors,
            'total_likes': total_likes,
            'avg_length': avg_length
        }
    }
