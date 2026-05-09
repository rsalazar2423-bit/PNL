"""
============================================================
Módulo: views/clustering_charts.py
============================================================
Visualizaciones del análisis de clustering y Visión 360º.

Responsabilidad Única:
    Generar gráficos de Plotly a partir de los datos numéricos
    calculados por models/clustering.py.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.core.theme import (
    apply_corporate_layout, CORP_BG, CORP_GRID, CORP_PALETTE
)


def create_silhouette_chart(sil_scores: list, best_k: int) -> go.Figure:
    """
    Genera el gráfico de evaluación Silhouette con marcador del K óptimo.

    Args:
        sil_scores (list[tuple]): Lista de (K, score).
        best_k (int): K óptimo seleccionado.

    Returns:
        go.Figure: Línea con área sombreada.
    """
    print("   [CLUST] Renderizando gráfica Silhouette...")
    ks, ss = zip(*sil_scores)
    fig = go.Figure(data=go.Scatter(
        x=list(ks), y=list(ss), mode='lines+markers',
        line=dict(width=4, color='#38bdf8'),
        marker=dict(size=12, color='#10b981', line=dict(width=2, color='white')),
        fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.2)'
    ))
    fig.add_vline(x=best_k, line_dash="dash", line_color='#F59E0B',
                  annotation_text=f"K óptimo = {best_k}")
    return apply_corporate_layout(fig, "Análisis de Agrupación Óptima (Silhouette)")


def create_tsne_3d_chart(df: pd.DataFrame, tsne_coords, sample_idx: list) -> go.Figure:
    """
    Genera el cubo 3D del espacio latente t-SNE.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'cluster' y 'text_clean'.
        tsne_coords (ndarray): Coordenadas t-SNE de forma (n_samples, 3).
        sample_idx (list[int]): Índices de las muestras en el DataFrame.

    Returns:
        go.Figure: Scatter 3D interactivo coloreado por cluster.
    """
    print("   [CLUST] Renderizando t-SNE 3D...")
    tsne_df = pd.DataFrame({
        'X': tsne_coords[:, 0], 'Y': tsne_coords[:, 1], 'Z': tsne_coords[:, 2],
        'cluster': [str(df.iloc[i]['cluster']) for i in sample_idx],
        'text': [df.iloc[i]['text_clean'][:100] for i in sample_idx]
    })

    fig = px.scatter_3d(
        tsne_df, x='X', y='Y', z='Z', color='cluster',
        hover_data=['text'], color_discrete_sequence=CORP_PALETTE,
        title=f"Espacio Latente 3D (t-SNE) - {len(sample_idx)} muestras"
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey'), opacity=0.8))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=CORP_BG,
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor=CORP_GRID),
            yaxis=dict(showbackground=False, gridcolor=CORP_GRID),
            zaxis=dict(showbackground=False, gridcolor=CORP_GRID),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, b=0, t=50), height=800
    )
    return fig


def create_cluster_sizes_chart(df: pd.DataFrame) -> go.Figure:
    """
    Genera barras con la distribución de tamaño por cluster.

    Args:
        df (pd.DataFrame): DataFrame con columna 'cluster'.

    Returns:
        go.Figure: Barras verticales con colorscale Viridis.
    """
    print("   [CLUST] Ensamblando barras de tamaño de clusters...")
    sizes = df['cluster'].value_counts().sort_index()
    labels = [f"Cluster {i}" for i in sizes.index]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=sizes.values,
        marker=dict(color=sizes.values, colorscale='Viridis',
                    line=dict(color='white', width=1.5)),
        text=sizes.values, textposition='auto'
    )])
    return apply_corporate_layout(fig, "Distribución Volumétrica por Cluster")


def create_topics_chart(lda_topics: list) -> go.Figure | None:
    """
    Genera la matriz de términos LDA como barras agrupadas.

    Args:
        lda_topics (list[dict]): Temas descubiertos por LDA.

    Returns:
        go.Figure | None: Barras agrupadas o None si no hay datos.
    """
    print("   [CLUST] Generando matriz de términos LDA...")
    topics_data = []
    for topic in lda_topics:
        for word, weight in topic['words'][:8]:
            topics_data.append({'Tema': topic['label'][:30], 'Palabra': word, 'Peso': weight})

    if not topics_data:
        return None

    df_topics = pd.DataFrame(topics_data)
    fig = px.bar(df_topics, x='Peso', y='Palabra', color='Tema',
                 orientation='h', barmode='group', color_discrete_sequence=CORP_PALETTE)
    fig.update_traces(marker_line_width=1.5, opacity=0.9)
    fig = apply_corporate_layout(fig, "Matriz de Términos LDA")
    fig.update_layout(height=700)
    return fig


def create_sunburst_360(df: pd.DataFrame) -> go.Figure:
    """
    Genera el gráfico jerárquico Sunburst (Corpus → Temas → Sentimiento).

    Args:
        df (pd.DataFrame): DataFrame con columnas 'cluster' y 'sentiment'.

    Returns:
        go.Figure: Sunburst chart interactivo.
    """
    print("   [CLUST] Renderizando Visión 360º (Sunburst Chart)...")
    if 'sentiment' not in df.columns or 'cluster' not in df.columns:
        print("   ⚠️ Faltan columnas. Sunburst omitido.")
        return go.Figure()

    df_temp = df.copy()
    df_temp['cluster_label'] = df_temp['cluster'].apply(lambda x: f"Tema {x}")
    agg_df = df_temp.groupby(['cluster_label', 'sentiment']).size().reset_index(name='count')
    agg_df['root'] = 'Corpus Total'

    color_map = {'POS': '#10B981', 'NEG': '#EF4444', 'NEU': '#64748B', 'Corpus Total': '#0F172A'}

    fig = px.sunburst(
        agg_df, path=['root', 'cluster_label', 'sentiment'],
        values='count', color='sentiment', color_discrete_map=color_map,
        title="Visión 360º: Temas y Percepción Emocional"
    )
    fig.update_traces(
        textinfo="label+percent parent", insidetextorientation='radial',
        marker=dict(line=dict(color=CORP_BG, width=2))
    )
    fig = apply_corporate_layout(fig, "Visión 360º: Temas y Percepción Emocional")
    fig.update_layout(height=700, margin=dict(t=50, l=0, r=0, b=0))
    return fig


def generate_clustering_charts(df: pd.DataFrame, cluster_data: dict) -> dict:
    """
    Orquesta la generación de todos los gráficos de clustering.

    Args:
        df (pd.DataFrame): DataFrame con columna 'cluster'.
        cluster_data (dict): Datos calculados por models/clustering.compute_clusters().

    Returns:
        dict: Diccionario con gráficos Plotly y metadatos.
    """
    return {
        'best_k': cluster_data['best_k'],
        'silhouette_chart': create_silhouette_chart(
            cluster_data['silhouette_scores'], cluster_data['best_k']),
        'tsne_chart': create_tsne_3d_chart(
            df, cluster_data['tsne_3d_coords'], cluster_data['tsne_sample_idx']),
        'cluster_sizes_chart': create_cluster_sizes_chart(df),
        'topics_chart': create_topics_chart(cluster_data['lda_topics']),
        'sunburst_360': create_sunburst_360(df),
    }
