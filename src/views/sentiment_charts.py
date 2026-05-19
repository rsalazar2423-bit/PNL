"""
============================================================
Módulo: views/sentiment_charts.py
============================================================
Visualizaciones del análisis de sentimiento.

Responsabilidad Única:
    Generar gráficos de Plotly a partir del DataFrame con
    las columnas 'sentiment' y 'sentiment_score' ya calculadas.
"""

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.theme import (
    apply_corporate_layout, CORP_BG, CORP_GRID,
    SENTIMENT_COLORS, SENTIMENT_LABELS
)


def create_distribution_pie(parquet_path: str) -> go.Figure:
    """
    Genera un Pie Chart volumétrico con la proporción de sentimientos.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Donut chart con pull en la categoría dominante.
    """
    print("   [SENT] Generando Pie Chart de sentimientos...")
    query = f"SELECT sentiment, count(*) as count FROM '{parquet_path}' WHERE sentiment IS NOT NULL GROUP BY sentiment"
    counts_df = duckdb.sql(query).df()
    
    if counts_df.empty:
        return go.Figure()
        
    counts_df.set_index('sentiment', inplace=True)
    counts = counts_df['count']
    
    fig = go.Figure(data=[go.Pie(
        labels=[SENTIMENT_LABELS.get(k, k) for k in counts.index],
        values=counts.values,
        hole=0.4,
        marker=dict(
            colors=[SENTIMENT_COLORS.get(k, '#000') for k in counts.index],
            line=dict(color=CORP_BG, width=3)
        ),
        pull=[0.05, 0, 0]
    )])
    fig = apply_corporate_layout(fig, "Proporción de Sentimientos (Volumétrico)")
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig


def create_bars_chart(parquet_path: str) -> go.Figure:
    """
    Genera barras absolutas con el conteo por polaridad.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Barras verticales coloreadas por sentimiento.
    """
    print("   [SENT] Generando barras de volumen absoluto...")
    query = f"SELECT sentiment, count(*) as count FROM '{parquet_path}' WHERE sentiment IS NOT NULL GROUP BY sentiment"
    counts_df = duckdb.sql(query).df()
    
    if counts_df.empty:
        return go.Figure()
        
    counts_df.set_index('sentiment', inplace=True)
    counts = counts_df['count']
    
    fig = go.Figure(data=[go.Bar(
        x=[SENTIMENT_LABELS.get(k, k) for k in counts.index],
        y=counts.values,
        marker=dict(
            color=[SENTIMENT_COLORS.get(k, '#000') for k in counts.index],
            line=dict(color='white', width=2)
        ),
        text=counts.values, textposition='auto'
    )])
    fig = apply_corporate_layout(fig, "Volumen Absoluto por Polaridad")
    fig.update_layout(showlegend=False)
    return fig


def create_3d_scatter(parquet_path: str) -> go.Figure:
    """
    Genera un Scatter 3D: Confianza × Likes × Longitud de texto.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Dispersión 3D interactiva con muestreo.
    """
    print("   [SENT] Renderizando dispersión 3D...")
    query = f"SELECT sentiment, sentiment_score, likes, length(text_clean) as length, text_clean FROM '{parquet_path}' USING SAMPLE 1500"
    df_3d_sample = duckdb.sql(query).df()

    if df_3d_sample.empty:
        return go.Figure()

    fig = px.scatter_3d(
        df_3d_sample,
        x='sentiment_score', y='likes', z='length',
        color='sentiment', color_discrete_map=SENTIMENT_COLORS,
        size='likes', size_max=40, opacity=0.8,
        hover_data=['text_clean'],
        title="Dispersión 3D: Sentimiento vs Impacto vs Longitud"
    )
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=CORP_BG,
        scene=dict(
            xaxis_title="Confianza", yaxis_title="Likes", zaxis_title="Longitud",
            xaxis=dict(gridcolor=CORP_GRID, showbackground=False),
            yaxis=dict(gridcolor=CORP_GRID, showbackground=False, type='log'),
            zaxis=dict(gridcolor=CORP_GRID, showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=50), height=700
    )
    return fig


def create_emotion_chart(parquet_path: str) -> go.Figure:
    """
    Genera un gráfico de barras con la distribución de emociones detectadas.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Gráfico de barras horizontales estilizado.
    """
    print("   [SENT] Generando gráfico de emociones...")
    query = f"SELECT emotion, count(*) as count FROM '{parquet_path}' WHERE emotion IS NOT NULL GROUP BY emotion"
    counts_df = duckdb.sql(query).df()
    
    if counts_df.empty:
        return go.Figure()
        
    counts_df.set_index('emotion', inplace=True)
    counts = counts_df['count']
    
    # Paleta de colores para emociones
    emotion_colors = {
        'others': '#94a3b8', 'joy': '#facc15', 'sadness': '#38bdf8',
        'anger': '#ef4444', 'surprise': '#a78bfa', 'disgust': '#fb923c',
        'fear': '#818cf8'
    }
    
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        marker=dict(
            color=[emotion_colors.get(k, '#94a3b8') for k in counts.index],
            line=dict(color='white', width=1)
        ),
        text=counts.values,
        textposition='auto'
    ))
    
    fig = apply_corporate_layout(fig, "Distribución Granular de Emociones")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def generate_sentiment_charts(parquet_path: str) -> dict:
    """
    Orquesta la generación de todos los gráficos de sentimiento usando DuckDB.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        dict: Diccionario con gráficos Plotly y Top 10 comentarios.
    """
    print("   [SENT] Construyendo visualizaciones de sentimiento [DuckDB]...")
    query_counts = f"SELECT sentiment, count(*) as count FROM '{parquet_path}' WHERE sentiment IS NOT NULL GROUP BY sentiment"
    counts_df = duckdb.sql(query_counts).df()
    
    if not counts_df.empty:
        counts_df.set_index('sentiment', inplace=True)
        counts = counts_df['count'].to_dict()
    else:
        counts = {}

    query_pos = f"SELECT text, likes, sentiment_score FROM '{parquet_path}' WHERE sentiment = 'POS' ORDER BY sentiment_score DESC LIMIT 10"
    top_pos = duckdb.sql(query_pos).df().to_dict('records')
    
    query_neg = f"SELECT text, likes, sentiment_score FROM '{parquet_path}' WHERE sentiment = 'NEG' ORDER BY sentiment_score DESC LIMIT 10"
    top_neg = duckdb.sql(query_neg).df().to_dict('records')

    return {
        'distribution_chart': create_distribution_pie(parquet_path),
        'bars_chart': create_bars_chart(parquet_path),
        'likes_sentiment_chart': create_3d_scatter(parquet_path),
        'emotion_chart': create_emotion_chart(parquet_path),
        'sentiment_counts': counts,
        'top_positive': top_pos,
        'top_negative': top_neg,
    }
