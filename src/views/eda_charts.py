"""
============================================================
Módulo: views/eda_charts.py
============================================================
Visualizaciones del Análisis Exploratorio de Datos (EDA).

Responsabilidad Única:
    Generar todos los gráficos de Plotly para la pestaña EDA
    a partir de un DataFrame ya preprocesado.
    NO ejecuta cálculos de modelo.
"""

import io
import re
import base64
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.theme import apply_corporate_layout, CORP_BG


def create_length_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Genera un histograma volumétrico con la distribución de longitud de caracteres.

    Args:
        df (pd.DataFrame): DataFrame con columna 'text_clean'.

    Returns:
        go.Figure: Histograma con línea de mediana.
    """
    print("   [EDA] Generando histograma volumétrico de longitudes...")
    lengths = df['text_clean'].str.len()
    median_len = lengths.median()

    fig = go.Figure(go.Histogram(
        x=lengths, nbinsx=50,
        marker=dict(color='#38bdf8', line=dict(color='white', width=1)),
        opacity=0.8
    ))
    fig.add_vline(
        x=median_len, line_dash="dash", line_color='#F59E0B',
        annotation_text=f"Mediana: {median_len:.0f} chars"
    )
    return apply_corporate_layout(fig, "Distribución de Longitud de Comentarios (Volumétrico)")


def create_top_authors_chart(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Genera barras horizontales con los autores más activos.

    Args:
        df (pd.DataFrame): DataFrame con columna 'author'.
        top_n (int): Número de autores a mostrar. Default: 20.

    Returns:
        go.Figure: Barras horizontales con sombreado de color.
    """
    print(f"   [EDA] Calculando top {top_n} autores más activos...")
    top_authors = df['author'].value_counts().head(top_n)
    fig = go.Figure(go.Bar(
        x=top_authors.values, y=top_authors.index, orientation='h',
        marker=dict(color=top_authors.values, colorscale='Tealgrn',
                    line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, f"Top {top_n} Autores Más Activos")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig


def create_likes_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Genera la distribución de likes en escala logarítmica.

    Args:
        df (pd.DataFrame): DataFrame con columna 'likes'.

    Returns:
        go.Figure: Histograma en escala log.
    """
    print("   [EDA] Procesando distribución de impacto (Likes)...")
    df_with_likes = df[df['likes'] > 0]
    fig = go.Figure(go.Histogram(
        x=df_with_likes['likes'], nbinsx=50,
        marker=dict(color='#8b5cf6', line=dict(color='white', width=1))
    ))
    fig = apply_corporate_layout(fig, "Impacto: Distribución de Likes")
    fig.update_layout(yaxis_type="log")
    return fig


def create_wordcloud_b64(df: pd.DataFrame) -> str:
    """
    Genera una nube de palabras como string Base64.

    Args:
        df (pd.DataFrame): DataFrame con columna 'lemmas_text'.

    Returns:
        str: Imagen PNG codificada en Base64.
    """
    print("   [EDA] Renderizando Nube de Palabras en memoria (Base64)...")
    all_text = ' '.join(df['lemmas_text'].dropna())
    wc = WordCloud(
        width=1200, height=600, background_color=CORP_BG,
        colormap='cool', max_words=150,
        contour_width=2, contour_color='#38bdf8', min_font_size=8
    ).generate(all_text)

    fig_wc, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig_wc.patch.set_facecolor(CORP_BG)

    buf = io.BytesIO()
    fig_wc.savefig(buf, format='png', bbox_inches='tight', facecolor=CORP_BG, dpi=150)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig_wc)
    return img_b64


def create_top_words_chart(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """
    Genera un Scatter 3D con la dispersión de los términos más frecuentes.

    Args:
        df (pd.DataFrame): DataFrame con columna 'lemmas'.
        top_n (int): Número de términos. Default: 30.

    Returns:
        go.Figure: Gráfico 3D interactivo.
    """
    print(f"   [EDA] Renderizando Dispersión 3D de términos (Top {top_n})...")
    all_lemmas = [l for lemmas in df['lemmas'] for l in lemmas]
    word_counts = Counter(all_lemmas).most_common(top_n)
    words, counts = zip(*word_counts)

    df_words = pd.DataFrame({
        'Término': words, 'Frecuencia': counts,
        'Longitud': [len(w) for w in words]
    })

    fig = px.scatter_3d(
        df_words, x='Longitud', y='Frecuencia', z='Frecuencia',
        color='Frecuencia', size='Frecuencia', text='Término',
        color_continuous_scale='Plasma',
        title=f"Dispersión 3D de Términos (Top {top_n})"
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=CORP_BG,
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='#1F2937'),
            yaxis=dict(showbackground=False, gridcolor='#1F2937'),
            zaxis=dict(showbackground=False, gridcolor='#1F2937')
        ),
        margin=dict(l=0, r=0, b=0, t=50), height=600
    )
    return fig


def _parse_youtube_relative_date(text: str) -> str:
    """
    Convierte una cadena de fecha relativa de YouTube a una categoría de antigüedad.

    Args:
        text (str): Cadena como 'hace 2 semanas (editado)'.

    Returns:
        str: Categoría ('0-6h', '6-24h', '1-3 días', etc.).
    """
    if not isinstance(text, str):
        return 'Desconocido'
    text = text.lower().replace('(editado)', '').strip()
    match = re.search(r'hace\s+(\d+)\s+(segundo|minuto|hora|día|dia|semana|mes|año|ano)', text)
    if not match:
        return 'Desconocido'
    cantidad = int(match.group(1))
    unidad = match.group(2).replace('día', 'dia').replace('año', 'ano')

    if unidad in ('segundo', 'minuto'):
        return '0-6h'
    elif unidad == 'hora':
        return '0-6h' if cantidad <= 6 else '6-24h'
    elif unidad == 'dia':
        return '1-3 días' if cantidad <= 3 else '3-7 días'
    elif unidad == 'semana':
        return '1-4 sem'
    elif unidad == 'mes':
        return '1+ mes'
    elif unidad == 'ano':
        return '1+ año'
    return 'Desconocido'


def create_temporal_chart(df: pd.DataFrame) -> go.Figure:
    """
    Genera barras de distribución de comentarios por antigüedad.

    Args:
        df (pd.DataFrame): DataFrame con columna 'published_at'.

    Returns:
        go.Figure: Barras con gradiente por categoría temporal.
    """
    print("   [EDA] Generando análisis de distribución temporal...")
    if 'published_at' not in df.columns:
        return go.Figure()

    df_time = df.copy()
    df_time['antiguedad'] = df_time['published_at'].apply(_parse_youtube_relative_date)

    orden = ['0-6h', '6-24h', '1-3 días', '3-7 días', '1-4 sem', '1+ mes', '1+ año']
    cats = [c for c in orden if c in df_time['antiguedad'].values]
    counts = df_time['antiguedad'].value_counts().reindex(cats).fillna(0).astype(int)

    colores = ['#10B981', '#38BDF8', '#818CF8', '#A78BFA', '#F59E0B', '#EF4444', '#991B1B']

    fig = go.Figure(go.Bar(
        x=counts.index.tolist(), y=counts.values.tolist(),
        marker=dict(color=colores[:len(cats)], line=dict(color='white', width=1.5)),
        text=counts.values.tolist(), textposition='auto'
    ))
    fig = apply_corporate_layout(fig, "Distribución Temporal: Antigüedad de Comentarios")
    fig.update_layout(xaxis_title="Antigüedad", yaxis_title="Cantidad de Comentarios", height=400)
    return fig


def generate_eda_plots(df: pd.DataFrame) -> dict:
    """
    Función orquestadora del EDA.

    Args:
        df (pd.DataFrame): DataFrame preprocesado.

    Returns:
        dict: Estadísticas globales y figuras de Plotly.
    """
    print("\n" + "=" * 60)
    print("📊 PASO 2: Análisis Exploratorio de Datos (EDA)")
    print("=" * 60)

    print("   [EDA] Calculando métricas globales...")
    stats = {
        'total_comments': len(df),
        'unique_authors': df['author'].nunique(),
        'avg_length': df['text_clean'].str.len().mean(),
        'total_likes': int(df['likes'].sum()),
    }

    plots = {
        'length_hist': create_length_histogram(df),
        'top_authors': create_top_authors_chart(df),
        'likes_dist': create_likes_distribution(df),
        'wordcloud_b64': create_wordcloud_b64(df),
        'top_words': create_top_words_chart(df),
        'temporal_chart': create_temporal_chart(df),
        'stats': stats
    }

    print("✅ [EDA] Construcción de gráficas finalizada.")
    return plots
