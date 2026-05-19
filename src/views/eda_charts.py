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

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core.theme import apply_corporate_layout, CORP_BG


def create_length_histogram(parquet_path: str) -> go.Figure:
    """
    Genera un histograma volumétrico con la distribución de longitud de caracteres.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Histograma con línea de mediana.
    """
    print("   [EDA] Generando histograma volumétrico de longitudes...")
    query = f"SELECT length(text_clean) as l FROM '{parquet_path}' WHERE text_clean IS NOT NULL"
    lengths = duckdb.sql(query).df()['l']
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


def create_top_authors_chart(parquet_path: str, top_n: int = 20) -> go.Figure:
    """
    Genera barras horizontales con los autores más activos.

    Args:
        parquet_path (str): Ruta al archivo parquet.
        top_n (int): Número de autores a mostrar. Default: 20.

    Returns:
        go.Figure: Barras horizontales con sombreado de color.
    """
    print(f"   [EDA] Calculando top {top_n} autores más activos...")
    query = f"SELECT author, COUNT(*) as c FROM '{parquet_path}' WHERE author IS NOT NULL GROUP BY author ORDER BY c DESC LIMIT {top_n}"
    top_authors = duckdb.sql(query).df()
    
    fig = go.Figure(go.Bar(
        x=top_authors['c'], y=top_authors['author'], orientation='h',
        marker=dict(color=top_authors['c'], colorscale='Tealgrn',
                    line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, f"Top {top_n} Autores Más Activos")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig


def create_likes_distribution(parquet_path: str) -> go.Figure:
    """
    Genera la distribución de likes en escala logarítmica.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Histograma en escala log.
    """
    print("   [EDA] Procesando distribución de impacto (Likes)...")
    query = f"SELECT likes FROM '{parquet_path}' WHERE likes > 0"
    df_with_likes = duckdb.sql(query).df()
    fig = go.Figure(go.Histogram(
        x=df_with_likes['likes'], nbinsx=50,
        marker=dict(color='#8b5cf6', line=dict(color='white', width=1))
    ))
    fig = apply_corporate_layout(fig, "Impacto: Distribución de Likes")
    fig.update_layout(yaxis_type="log")
    return fig


def create_wordcloud_b64(parquet_path: str) -> str:
    """
    Genera una nube de palabras como string Base64.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        str: Imagen PNG codificada en Base64.
    """
    print("   [EDA] Renderizando Nube de Palabras en memoria (Base64)...")
    query = f"SELECT lemmas_text FROM '{parquet_path}' WHERE lemmas_text IS NOT NULL"
    lemmas_df = duckdb.sql(query).df()
    all_text = ' '.join(lemmas_df['lemmas_text'])
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


def create_top_words_chart(parquet_path: str, top_n: int = 30) -> go.Figure:
    """
    Genera un Scatter 3D con la dispersión de los términos más frecuentes.

    Args:
        parquet_path (str): Ruta al archivo parquet.
        top_n (int): Número de términos. Default: 30.

    Returns:
        go.Figure: Gráfico 3D interactivo.
    """
    print(f"   [EDA] Renderizando Dispersión 3D de términos (Top {top_n})...")
    # Extraemos todos los lemmas_text, los unimos y contamos
    query = f"SELECT lemmas_text FROM '{parquet_path}' WHERE lemmas_text IS NOT NULL"
    lemmas_df = duckdb.sql(query).df()
    
    all_lemmas = []
    for text in lemmas_df['lemmas_text']:
        if isinstance(text, str):
            all_lemmas.extend(text.split())
            
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


def create_temporal_chart(parquet_path: str) -> go.Figure:
    """
    Genera barras de distribución de comentarios por antigüedad.

    Args:
        parquet_path (str): Ruta al archivo parquet.

    Returns:
        go.Figure: Barras con gradiente por categoría temporal.
    """
    print("   [EDA] Generando análisis de distribución temporal...")
    query = f"SELECT published_at FROM '{parquet_path}' WHERE published_at IS NOT NULL"
    try:
        df_time = duckdb.sql(query).df()
    except Exception:
        return go.Figure()
        
    if df_time.empty:
        return go.Figure()

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


def generate_eda_plots(parquet_path: str) -> dict:
    """
    Función orquestadora del EDA.

    Args:
        parquet_path (str): Ruta al archivo parquet preprocesado.

    Returns:
        dict: Estadísticas globales y figuras de Plotly.
    """
    print("\n" + "=" * 60)
    print("📊 PASO 2: Análisis Exploratorio de Datos (EDA) [DuckDB]")
    print("=" * 60)

    print("   [EDA] Calculando métricas globales...")
    query = f"SELECT count(*) as tc, count(DISTINCT author) as ua, avg(length(text_clean)) as al, sum(likes) as tl FROM '{parquet_path}'"
    res = duckdb.sql(query).fetchone()
    
    stats = {
        'total_comments': res[0],
        'unique_authors': res[1],
        'avg_length': res[2] if res[2] else 0,
        'total_likes': int(res[3]) if res[3] else 0,
    }

    plots = {
        'length_hist': create_length_histogram(parquet_path),
        'top_authors': create_top_authors_chart(parquet_path),
        'likes_dist': create_likes_distribution(parquet_path),
        'wordcloud_b64': create_wordcloud_b64(parquet_path),
        'top_words': create_top_words_chart(parquet_path),
        'temporal_chart': create_temporal_chart(parquet_path),
        'stats': stats
    }

    print("✅ [EDA] Construcción de gráficas finalizada.")
    return plots
