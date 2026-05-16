"""
============================================================
Módulo: core/theme.py
============================================================
Sistema de diseño corporativo centralizado (Dark Premium).

Contiene todas las constantes visuales, paletas de color y la
función de aplicación de layout para gráficos Plotly. Al centralizar
estos valores, cualquier cambio de identidad visual se realiza
en un único lugar sin necesidad de modificar cada módulo de gráficos.

Uso:
    >>> from src.core.theme import apply_corporate_layout, CORP_BG
    >>> fig = apply_corporate_layout(fig, "Mi Título")
"""

import plotly.graph_objects as go
import gradio as gr



# ─────────────────────────────────────────────────────────────
# PALETA DE COLORES CORPORATIVA
# ─────────────────────────────────────────────────────────────

CORP_BG = '#0B0F19'
"""str: Color de fondo principal del dashboard (azul navy ultra-oscuro)."""

CORP_GRID = '#1F2937'
"""str: Color de las líneas de la grilla en los gráficos."""

CORP_PALETTE = [
    '#38bdf8', '#818cf8', '#a78bfa', '#f472b6',
    '#fb923c', '#facc15', '#4ade80', '#2dd4bf',
    '#94a3b8', '#e2e8f0'
]
"""list[str]: Paleta de 10 colores para series de datos en gráficos multi-categoría."""

# Colores semánticos para sentimientos
COLOR_POS = '#10B981'
"""str: Verde esmeralda para sentimiento positivo."""

COLOR_NEG = '#EF4444'
"""str: Rojo para sentimiento negativo."""

COLOR_NEU = '#94A3B8'
"""str: Gris pizarra para sentimiento neutro."""

SENTIMENT_COLORS = {
    'POS': COLOR_POS,
    'NEG': COLOR_NEG,
    'NEU': COLOR_NEU,
}
"""dict: Mapa de etiqueta de sentimiento → color hex."""

SENTIMENT_LABELS = {
    'POS': 'Positivo',
    'NEG': 'Negativo',
    'NEU': 'Neutro',
}
"""dict: Mapa de etiqueta de sentimiento → nombre legible."""


# ─────────────────────────────────────────────────────────────
# FUNCIÓN CENTRAL DE LAYOUT
# ─────────────────────────────────────────────────────────────

def apply_corporate_layout(fig: go.Figure, title: str) -> go.Figure:
    """
    Aplica el diseño corporativo oscuro (Dark Premium) a un gráfico de Plotly.

    Configura el fondo, la tipografía, la grilla y los márgenes
    para que todos los gráficos del dashboard tengan una apariencia
    visual consistente y profesional.

    Args:
        fig (go.Figure): Objeto de figura de Plotly a estilizar.
        title (str): Título que se mostrará en la parte superior del gráfico.

    Returns:
        go.Figure: La misma figura con el layout corporativo aplicado.

    Example:
        >>> fig = go.Figure(go.Bar(x=[1,2], y=[3,4]))
        >>> fig = apply_corporate_layout(fig, "Ventas Trimestrales")
    """
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 20, 'family': 'Inter', 'color': '#F8FAFC'}
        },
        template='plotly_dark',
        paper_bgcolor=CORP_BG,
        plot_bgcolor=CORP_BG,
        font=dict(family="Inter", size=13, color="#CBD5E1"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor=CORP_GRID, zerolinecolor=CORP_GRID),
        yaxis=dict(showgrid=True, gridcolor=CORP_GRID, zerolinecolor=CORP_GRID)
    )
    return fig


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN GRADIO (UI)
# ─────────────────────────────────────────────────────────────

def get_gradio_theme():
    """Retorna el tema Dark Premium para la interfaz Gradio."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill=CORP_BG,
        body_text_color="#CBD5E1",
        block_background_fill="#111827",
        block_border_color="#1E293B",
        block_label_text_color="#94A3B8",
        block_title_text_color="#F8FAFC",
        input_background_fill="#1E293B",
        button_primary_background_fill="#2563EB",
        button_secondary_background_fill="#1E293B",
    )

def get_dark_mode_js():
    """Retorna el script JS para forzar el modo oscuro en el navegador."""
    return """
    function force_dark_mode() {
        document.body.classList.add('dark');
    }
    """
