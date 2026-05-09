"""
============================================================
Módulo: views/tfidf_charts.py
============================================================
Visualizaciones del análisis TF-IDF.

Responsabilidad Única:
    Generar gráficos de Plotly a partir de los datos numéricos
    calculados por models/tfidf.py.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.core.theme import apply_corporate_layout, CORP_BG, CORP_GRID


def create_top_terms_chart(top_terms: list) -> go.Figure:
    """
    Genera barras horizontales con los términos de mayor impacto TF-IDF.

    Args:
        top_terms (list[tuple]): Lista de (término, score_promedio).

    Returns:
        go.Figure: Barras horizontales con gradiente de color.
    """
    print("   [TF-IDF] Generando barras de términos principales...")
    terms, scores = zip(*top_terms)
    fig = go.Figure(go.Bar(
        x=list(scores), y=list(terms), orientation='h',
        marker=dict(color=list(scores), colorscale='Plasma',
                    line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, f"Top {len(terms)} Términos por Relevancia (TF-IDF)")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=700)
    return fig


def create_bigrams_chart(bigrams: list) -> go.Figure:
    """
    Genera barras horizontales con los bigramas más frecuentes.

    Args:
        bigrams (list[tuple]): Lista de (bigrama, frecuencia).

    Returns:
        go.Figure: Barras horizontales.
    """
    print("   [TF-IDF] Renderizando gráfico de bigramas...")
    if not bigrams:
        return go.Figure()
    bg_labels, bg_counts = zip(*bigrams)
    fig = go.Figure(go.Bar(
        x=list(bg_counts), y=list(bg_labels), orientation='h',
        marker=dict(color='#38bdf8', line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, "Bigramas Más Frecuentes")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig


def create_trigrams_chart(trigrams: list) -> go.Figure:
    """
    Genera barras horizontales con los trigramas más frecuentes.

    Args:
        trigrams (list[tuple]): Lista de (trigrama, frecuencia).

    Returns:
        go.Figure: Barras horizontales.
    """
    print("   [TF-IDF] Renderizando gráfico de trigramas...")
    if not trigrams:
        return go.Figure()
    tg_labels, tg_counts = zip(*trigrams)
    fig = go.Figure(go.Bar(
        x=list(tg_counts), y=list(tg_labels), orientation='h',
        marker=dict(color='#a78bfa', line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, "Trigramas Más Frecuentes")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig


def generate_tfidf_charts(tfidf_data: dict) -> dict:
    """
    Orquesta la generación de todos los gráficos TF-IDF.

    Args:
        tfidf_data (dict): Datos calculados por models/tfidf.compute_tfidf().

    Returns:
        dict: Diccionario con gráficos Plotly y datos del modelo.
    """
    return {
        'tfidf_matrix': tfidf_data['tfidf_matrix'],
        'feature_names': tfidf_data['feature_names'],
        'top_terms_chart': create_top_terms_chart(tfidf_data['top_terms']),
        'bigrams_chart': create_bigrams_chart(tfidf_data['bigrams']),
        'trigrams_chart': create_trigrams_chart(tfidf_data['trigrams']),
    }
