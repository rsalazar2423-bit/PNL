"""
============================================================
Módulo: views/clustering_charts.py
============================================================
Visualizaciones del análisis de clustering semántico.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from src.core.theme import (
    apply_corporate_layout, CORP_BG, CORP_GRID, CORP_PALETTE
)

def create_silhouette_chart(sil_scores: list, best_k: int) -> go.Figure:
    """
    Genera el gráfico de evaluación Silhouette.
    """
    if not sil_scores:
        # Retornar gráfico vacío con mensaje si no hay datos
        fig = go.Figure()
        fig.add_annotation(text="Datos de silueta no disponibles en esta versión", showarrow=False)
        return apply_corporate_layout(fig, "Análisis de Agrupación (Silhouette)")

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

def create_tsne_3d_chart(df: pd.DataFrame, tsne_coords, sample_idx: list, cluster_labels: dict) -> go.Figure:
    """
    Genera el cubo 3D del espacio latente t-SNE con etiquetas semánticas.
    """
    print("   [CLUST] Renderizando t-SNE 3D Semántico...")
    
    tsne_df = pd.DataFrame({
        'X': tsne_coords[:, 0], 'Y': tsne_coords[:, 1], 'Z': tsne_coords[:, 2],
        'cluster': [cluster_labels.get(df.iloc[i]['cluster'], f"Tema {df.iloc[i]['cluster']}") for i in sample_idx],
        'text': [df.iloc[i]['text_clean'][:100] for i in sample_idx]
    })

    fig = px.scatter_3d(
        tsne_df, x='X', y='Y', z='Z', color='cluster',
        hover_data=['text'], color_discrete_sequence=CORP_PALETTE,
        title=f"Espacio Semántico 3D (t-SNE) - {len(sample_idx)} muestras"
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white'), opacity=0.8))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor=CORP_BG,
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor=CORP_GRID, title="Semántica X"),
            yaxis=dict(showbackground=False, gridcolor=CORP_GRID, title="Semántica Y"),
            zaxis=dict(showbackground=False, gridcolor=CORP_GRID, title="Semántica Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=50), height=800
    )
    return fig

def create_cluster_sizes_chart(df: pd.DataFrame, cluster_labels: dict) -> go.Figure:
    """
    Genera barras con la distribución de tamaño por cluster.
    """
    sizes = df['cluster'].value_counts().sort_index()
    labels = [cluster_labels.get(i, f"Tema {i}")[:40] + "..." for i in sizes.index]

    fig = go.Figure(data=[go.Bar(
        x=labels, y=sizes.values,
        marker=dict(color=sizes.values, colorscale='Viridis',
                    line=dict(color='white', width=1.5)),
        text=sizes.values, textposition='auto'
    )])
    fig = apply_corporate_layout(fig, "Distribución de Temas Detectados")
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_sunburst_360(df: pd.DataFrame, cluster_labels: dict) -> go.Figure:
    """
    Genera el gráfico jerárquico Sunburst (Corpus → Temas → Sentimiento).
    """
    if 'sentiment' not in df.columns or 'cluster' not in df.columns:
        return go.Figure()

    df_temp = df.copy()
    df_temp['cluster_label'] = df_temp['cluster'].apply(lambda x: cluster_labels.get(x, f"Tema {x}")[:30])
    
    agg_df = df_temp.groupby(['cluster_label', 'sentiment']).size().reset_index(name='count')
    agg_df['root'] = 'Audiencia Total'

    color_map = {'POS': '#10B981', 'NEG': '#EF4444', 'NEU': '#64748B', 'Audiencia Total': '#0F172A'}

    fig = px.sunburst(
        agg_df, path=['root', 'cluster_label', 'sentiment'],
        values='count', color='sentiment', color_discrete_map=color_map,
    )
    fig.update_traces(
        textinfo="label+percent parent",
        marker=dict(line=dict(color=CORP_BG, width=2))
    )
    fig = apply_corporate_layout(fig, "Visión 360º: Temas y Percepción")
    fig.update_layout(height=700, margin=dict(t=50, l=0, r=0, b=0))
    return fig

def generate_clustering_charts(df: pd.DataFrame, cluster_data: dict) -> dict:
    """
    Orquesta la generación de todos los gráficos de clustering.
    """
    cluster_labels = cluster_data.get('cluster_labels', {})
    
    return {
        'best_k': cluster_data['best_k'],
        'silhouette_chart': create_silhouette_chart(
            cluster_data.get('silhouette_scores', []), cluster_data['best_k']),
        'tsne_chart': create_tsne_3d_chart(
            df, cluster_data['tsne_3d_coords'], cluster_data['tsne_sample_idx'], cluster_labels),
        'cluster_sizes_chart': create_cluster_sizes_chart(df, cluster_labels),
        'sunburst_360': create_sunburst_360(df, cluster_labels),
    }
