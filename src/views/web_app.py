"""
============================================================
Módulo: web_app.py
============================================================
Construye la interfaz web interactiva usando Gradio Blocks.
Diseño puramente declarativo; la lógica reside en controllers/.
"""

import gradio as gr
from src.core.theme import get_gradio_theme, get_dark_mode_js
from src.controllers.dashboard_controller import load_dashboard_ui, respond, export_data

def create_web_app() -> tuple:
    """
    Construye y retorna la aplicación Gradio completa.
    """
    
    with gr.Blocks(
        title="Panel de Inteligencia Corporativo", 
        theme=get_gradio_theme(), 
        js=get_dark_mode_js()
    ) as app:

        gr.Markdown("# Panel de Inteligencia de Opinión — Análisis NLP\n### Estudio de Audiencia: Entrevista a Juan Daniel Oviedo")

        rag_state = gr.State(None)

        # ── PANTALLA 1: Carga ──
        with gr.Column(visible=True) as loading_screen:
            gr.Markdown("### ⚙️ El sistema de IA requiere inicializarse.")
            btn_start = gr.Button("🚀 Iniciar Procesamiento NLP", variant="primary", size="lg")
            loading_status = gr.Textbox(label="Estado del Servidor", value="Esperando inicio...", interactive=False)

        # ── PANTALLA 2: Dashboard ──
        with gr.Column(visible=False) as dashboard_screen:
            with gr.Tabs():
                with gr.Tab("Resumen Ejecutivo"):
                    with gr.Row():
                        kpi_1 = gr.Number(label="Comentarios Analizados")
                        kpi_2 = gr.Number(label="Autores Únicos")
                        kpi_3 = gr.Number(label="Interacciones (Likes)")
                        kpi_4 = gr.Number(label="Longitud Promedio")
                    btn_export = gr.DownloadButton("📥 Descargar Datos Procesados (CSV)", size="lg", variant="secondary")
                    wc_html = gr.HTML()

                with gr.Tab("Análisis Exploratorio"):
                    plot_hist = gr.Plot()
                    plot_temporal = gr.Plot()
                    with gr.Row():
                        plot_auth = gr.Plot()
                        plot_likes = gr.Plot()
                    plot_words = gr.Plot()

                with gr.Tab("Relevancia Semántica"):
                    plot_tf_top = gr.Plot()
                    with gr.Row():
                        plot_tf_bi = gr.Plot()
                        plot_tf_tri = gr.Plot()

                with gr.Tab("Mapeo de Entidades"):
                    plot_ner_top = gr.Plot()
                    plot_ner_type = gr.Plot()
                    plot_ner_pos = gr.Plot()

                with gr.Tab("Análisis de Percepción"):
                    with gr.Row():
                        plot_sent_dist = gr.Plot()
                        plot_sent_bar = gr.Plot()
                    plot_sent_emo = gr.Plot()
                    plot_sent_3d = gr.Plot()

                with gr.Tab("Visión 360º (Jerarquía)"):
                    plot_sunburst = gr.Plot()

                with gr.Tab("Temas Latentes (Clustering)"):
                    txt_cluster = gr.Markdown("")
                    plot_clust_sil = gr.Plot()
                    plot_clust_tsne = gr.Plot()
                    plot_clust_size = gr.Plot()

                with gr.Tab("Asistente Cognitivo"):
                    rag_mode = gr.Radio(["rustico", "ia"], label="Modo de Búsqueda", value="rustico")
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(label="Pregunta al modelo:")
                    clear = gr.Button("Limpiar")

        # ── EVENTOS ──
        msg.submit(respond, [msg, chatbot, rag_state, rag_mode], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        btn_export.click(fn=export_data, inputs=[], outputs=btn_export)

        outputs = [
            loading_screen, dashboard_screen, loading_status, rag_state,
            kpi_1, kpi_2, kpi_3, kpi_4, wc_html,
            plot_hist, plot_temporal, plot_auth, plot_likes, plot_words,
            plot_tf_top, plot_tf_bi, plot_tf_tri,
            plot_ner_top, plot_ner_type, plot_ner_pos,
            plot_sent_dist, plot_sent_bar, plot_sent_emo, plot_sent_3d,
            txt_cluster, plot_clust_sil, plot_clust_tsne, plot_clust_size,
            plot_sunburst
        ]
        
    return app, btn_start, outputs
