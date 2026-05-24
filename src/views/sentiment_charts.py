"""
============================================================
Módulo: views/sentiment_charts.py
============================================================
Visualizaciones del análisis de sentimiento.

Responsabilidad Única:
    Generar gráficos de Plotly a partir de la base de datos DuckDB.
"""

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.theme import (
    apply_corporate_layout, CORP_BG, CORP_GRID,
    SENTIMENT_COLORS, SENTIMENT_LABELS
)

# Paleta de colores unificada para emociones
EMOTION_COLORS = {
    'others': '#94a3b8', 'joy': '#facc15', 'sadness': '#38bdf8',
    'anger': '#ef4444', 'surprise': '#a78bfa', 'disgust': '#fb923c',
    'fear': '#818cf8'
}


def _get_sentiment_counts(parquet_path: str) -> pd.DataFrame:
    """
    Helper para obtener el conteo de sentimientos consolidado de DuckDB.
    """
    query = f"SELECT sentiment, count(*) as count FROM '{parquet_path}' WHERE sentiment IS NOT NULL GROUP BY sentiment"
    try:
        return duckdb.sql(query).df()
    except Exception as e:
        print(f"   [SENT-DB] Error al consultar sentimientos: {e}")
        return pd.DataFrame()


def create_distribution_pie(parquet_path: str) -> go.Figure:
    """
    Genera un Pie Chart volumétrico con la proporción de sentimientos.
    """
    print("   [SENT] Generando Pie Chart de sentimientos...")
    counts_df = _get_sentiment_counts(parquet_path)
    
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
    """
    print("   [SENT] Generando barras de volumen absoluto...")
    counts_df = _get_sentiment_counts(parquet_path)
    
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
    """
    print("   [SENT] Renderizando dispersión 3D...")
    query = f"SELECT sentiment, sentiment_score, likes, length(text_clean) as length, text_clean FROM '{parquet_path}' USING SAMPLE 1500"
    try:
        df_3d_sample = duckdb.sql(query).df()
    except Exception:
        df_3d_sample = pd.DataFrame()

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
    """
    print("   [SENT] Generando gráfico de emociones...")
    query = f"SELECT emotion, count(*) as count FROM '{parquet_path}' WHERE emotion IS NOT NULL GROUP BY emotion"
    try:
        counts_df = duckdb.sql(query).df()
    except Exception:
        counts_df = pd.DataFrame()
    
    if counts_df.empty:
        return go.Figure()
        
    counts_df.set_index('emotion', inplace=True)
    counts = counts_df['count']
    
    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        marker=dict(
            color=[EMOTION_COLORS.get(k, '#94a3b8') for k in counts.index],
            line=dict(color='white', width=1)
        ),
        text=counts.values,
        textposition='auto'
    ))
    
    fig = apply_corporate_layout(fig, "Distribución Granular de Emociones")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def create_confidence_distribution_chart(parquet_path: str) -> dict:
    """
    Genera histogramas de distribución de confianza para sentimientos y emociones.
    """
    print("   [SENT] Generando gráficos de distribución de confianza...")
    query = f"SELECT sentiment_score, emotion_score FROM '{parquet_path}' WHERE sentiment_score IS NOT NULL AND emotion_score IS NOT NULL"
    try:
        df_conf = duckdb.sql(query).df()
    except Exception:
        df_conf = pd.DataFrame()

    if df_conf.empty:
        fig_sent = go.Figure()
        fig_emo = go.Figure()
        return {'sentiment_confidence': fig_sent, 'emotion_confidence': fig_emo}

    fig_sent = go.Figure(go.Histogram(
        x=df_conf['sentiment_score'] * 100, nbinsx=30,
        marker=dict(color='#10b981', line=dict(color='white', width=1)),
        opacity=0.8
    ))
    fig_sent = apply_corporate_layout(fig_sent, "Distribución de Confianza: Sentimientos")
    fig_sent.update_layout(xaxis_title="Confianza (%)", yaxis_title="Cantidad de Comentarios")

    fig_emo = go.Figure(go.Histogram(
        x=df_conf['emotion_score'] * 100, nbinsx=30,
        marker=dict(color='#8b5cf6', line=dict(color='white', width=1)),
        opacity=0.8
    ))
    fig_emo = apply_corporate_layout(fig_emo, "Distribución de Confianza: Emociones")
    fig_emo.update_layout(xaxis_title="Confianza (%)", yaxis_title="Cantidad de Comentarios")

    return {'sentiment_confidence': fig_sent, 'emotion_confidence': fig_emo}


def create_correlation_heatmap(parquet_path: str) -> go.Figure:
    """
    Genera un Heatmap de correlación entre variables cuantitativas y el sentimiento.
    """
    print("   [SENT] Generando Heatmap de correlación...")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"   [SENT] Error al leer parquet para correlación: {e}")
        return go.Figure()

    # Mapear sentimiento a numérico (POS = 1, NEU = 0, NEG = -1)
    sentiment_map = {'POS': 1, 'NEU': 0, 'NEG': -1}
    df['sentiment_numeric'] = df['sentiment'].map(sentiment_map).fillna(0)

    # Calcular longitud de comentarios en caracteres y palabras
    df['comment_length'] = df['text'].fillna('').str.len()
    df['word_count'] = df['text_clean'].fillna('').str.split().str.len()

    cols = ['likes', 'comment_length', 'word_count', 'sentiment_numeric', 'sentiment_score', 'emotion_score']
    labels = {
        'likes': 'Likes',
        'comment_length': 'Longitud (Caract.)',
        'word_count': 'Cant. Palabras',
        'sentiment_numeric': 'Sentimiento (Num)',
        'sentiment_score': 'Conf. Sentimiento',
        'emotion_score': 'Conf. Emoción'
    }

    corr_df = df[cols].corr()
    corr_df.rename(columns=labels, index=labels, inplace=True)

    fig = px.imshow(
        corr_df,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        labels=dict(color="Correlación")
    )
    
    fig = apply_corporate_layout(fig, "Matriz de Correlación de Variables")
    fig.update_xaxes(side="bottom")
    return fig


def generate_sentiment_charts(parquet_path: str) -> dict:
    """
    Orquesta la generación de todos los gráficos de sentimiento usando DuckDB.
    """
    print("   [SENT] Construyendo visualizaciones de sentimiento [DuckDB]...")
    
    counts_df = _get_sentiment_counts(parquet_path)
    if not counts_df.empty:
        counts_df.set_index('sentiment', inplace=True)
        counts = counts_df['count'].to_dict()
    else:
        counts = {}
 
    query_pos = f"SELECT text, likes, sentiment_score FROM '{parquet_path}' WHERE sentiment = 'POS' ORDER BY sentiment_score DESC LIMIT 10"
    try:
        top_pos = duckdb.sql(query_pos).df().to_dict('records')
    except Exception:
        top_pos = []
    
    query_neg = f"SELECT text, likes, sentiment_score FROM '{parquet_path}' WHERE sentiment = 'NEG' ORDER BY sentiment_score DESC LIMIT 10"
    try:
        top_neg = duckdb.sql(query_neg).df().to_dict('records')
    except Exception:
        top_neg = []

    conf_charts = create_confidence_distribution_chart(parquet_path)

    return {
        'distribution_chart': create_distribution_pie(parquet_path),
        'bars_chart': create_bars_chart(parquet_path),
        'likes_sentiment_chart': create_3d_scatter(parquet_path),
        'emotion_chart': create_emotion_chart(parquet_path),
        'sentiment_confidence': conf_charts['sentiment_confidence'],
        'emotion_confidence': conf_charts['emotion_confidence'],
        'correlation_chart': create_correlation_heatmap(parquet_path),
        'sentiment_counts': counts,
        'top_positive': top_pos,
        'top_negative': top_neg,
    }
