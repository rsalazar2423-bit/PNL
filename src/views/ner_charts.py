"""
============================================================
Módulo: views/ner_charts.py
============================================================
Visualizaciones del análisis NER y POS Tagging.

Responsabilidad Única:
    Generar gráficos de Plotly a partir de las frecuencias
    calculadas por models/ner.py.
"""

import plotly.graph_objects as go
from src.core.theme import apply_corporate_layout


def create_entities_chart(top_entities: list) -> go.Figure | None:
    """
    Genera barras horizontales con las entidades más mencionadas.

    Args:
        top_entities (list[tuple]): Lista de (entidad, frecuencia).

    Returns:
        go.Figure | None: Gráfico o None si no hay datos.
    """
    print("   [NER] Renderizando gráfico de entidades...")
    if not top_entities:
        return None
    names, counts = zip(*top_entities)
    fig = go.Figure(go.Bar(
        x=list(counts), y=list(names), orientation='h',
        marker=dict(color=list(counts), colorscale='Inferno',
                    line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, "Entidades Más Mencionadas (NER)")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig


def create_entity_types_chart(entity_type_counts: dict) -> go.Figure | None:
    """
    Genera un Pie Chart con la distribución de tipos de entidad.

    Args:
        entity_type_counts (dict): Conteo {PER: n, LOC: n, ORG: n, ...}.

    Returns:
        go.Figure | None: Pie chart o None si no hay datos.
    """
    print("   [NER] Generando Pie Chart de tipos de entidades...")
    if not entity_type_counts:
        return None
    labels = list(entity_type_counts.keys())
    counts = list(entity_type_counts.values())
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=counts, hole=0.3,
        marker=dict(line=dict(color='#0B0F19', width=2)),
        pull=[0.05] * len(labels)
    )])
    fig = apply_corporate_layout(fig, "Distribución de Tipos de Entidades")
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig


def create_pos_chart(top_pos_tags: list) -> go.Figure | None:
    """
    Genera barras horizontales con las categorías gramaticales.

    Args:
        top_pos_tags (list[tuple]): Lista de (tag, frecuencia).

    Returns:
        go.Figure | None: Gráfico o None si no hay datos.
    """
    print("   [NER] Creando Gráfico de Categorías Gramaticales...")
    if not top_pos_tags:
        return None
    tags, counts = zip(*top_pos_tags)
    fig = go.Figure(go.Bar(
        x=list(counts), y=list(tags), orientation='h',
        marker=dict(color=list(counts), colorscale='Viridis',
                    line=dict(color='white', width=1.5))
    ))
    fig = apply_corporate_layout(fig, "Categorías Gramaticales (Morfología)")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
    return fig


def generate_ner_charts(ner_data: dict) -> dict:
    """
    Orquesta la generación de todos los gráficos NER/POS.

    Args:
        ner_data (dict): Datos calculados por models/ner.extract_entities().

    Returns:
        dict: Diccionario con gráficos Plotly.
    """
    return {
        'top_entities_chart': create_entities_chart(ner_data['top_entities']),
        'entity_types_chart': create_entity_types_chart(ner_data['entity_type_counts']),
        'pos_chart': create_pos_chart(ner_data['top_pos_tags']),
    }
