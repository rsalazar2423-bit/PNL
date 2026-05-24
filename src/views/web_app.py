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

def _build_summary_tab() -> tuple:
    with gr.Tab("Resumen Ejecutivo"):
        with gr.Row():
            kpi_1 = gr.Number(label="Comentarios Analizados")
            kpi_2 = gr.Number(label="Autores Únicos")
            kpi_3 = gr.Number(label="Interacciones (Likes)")
            kpi_4 = gr.Number(label="Longitud Promedio")
        with gr.Row():
            kpi_nss = gr.Textbox(label="Net Sentiment Score (Reputación Neta)", interactive=False)
            kpi_nss_w = gr.Textbox(label="Reputación Ponderada por Likes (Impacto Real)", interactive=False)
            kpi_pol = gr.Textbox(label="Índice de Polarización de Audiencia", interactive=False)
        btn_export = gr.DownloadButton("📥 Descargar Datos Procesados (CSV)", size="lg", variant="secondary")
        wc_html = gr.HTML()
        gr.Markdown("### 📋 Diagnóstico de Marca & Recomendaciones Estratégicas para Oviedo")
        diagnostic_md = gr.Markdown()
    return kpi_1, kpi_2, kpi_3, kpi_4, kpi_nss, kpi_nss_w, kpi_pol, btn_export, wc_html, diagnostic_md

def _build_clustering_tab() -> tuple:
    with gr.Tab("Temas Latentes (Clustering)"):
        txt_cluster = gr.Markdown("")
        plot_clust_sil = gr.Plot()
        plot_clust_tsne = gr.Plot()
        plot_clust_size = gr.Plot()
    return txt_cluster, plot_clust_sil, plot_clust_tsne, plot_clust_size

def _build_perception_tab() -> tuple:
    with gr.Tab("Análisis de Percepción"):
        with gr.Row():
            plot_sent_dist = gr.Plot()
            plot_sent_bar = gr.Plot()
        plot_sent_emo = gr.Plot()
        plot_sent_3d = gr.Plot()
    return plot_sent_dist, plot_sent_bar, plot_sent_emo, plot_sent_3d

def _build_hierarchy_tab() -> gr.Plot:
    with gr.Tab("Visión 360º (Jerarquía)"):
        plot_sunburst = gr.Plot()
    return plot_sunburst

def _build_eda_tab() -> tuple:
    with gr.Tab("Análisis Exploratorio"):
        plot_hist = gr.Plot()
        plot_temporal = gr.Plot()
        with gr.Row():
            plot_auth = gr.Plot()
            plot_likes = gr.Plot()
        plot_words = gr.Plot()
    return plot_hist, plot_temporal, plot_auth, plot_likes, plot_words

def _build_semantic_tab() -> tuple:
    with gr.Tab("Relevancia Semántica"):
        plot_tf_top = gr.Plot()
        with gr.Row():
            plot_tf_bi = gr.Plot()
            plot_tf_tri = gr.Plot()
    return plot_tf_top, plot_tf_bi, plot_tf_tri

def _build_ner_tab() -> tuple:
    with gr.Tab("Mapeo de Entidades"):
        plot_ner_top = gr.Plot()
        plot_ner_type = gr.Plot()
        plot_ner_pos = gr.Plot()
    return plot_ner_top, plot_ner_type, plot_ner_pos

def _build_rag_tab() -> tuple:
    with gr.Tab("Asistente Cognitivo"):
        rag_mode = gr.Radio(["rustico", "ia"], label="Modo de Búsqueda", value="rustico")
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Pregunta al modelo:")
        clear = gr.Button("Limpiar")
    return rag_mode, chatbot, msg, clear

def _build_evaluation_tab() -> tuple:
    with gr.Tab("Evaluación y Métricas"):
        gr.Markdown("### 📊 Métricas de Calidad de Modelos NLP")
        with gr.Row():
            eval_kpi_1 = gr.Textbox(label="Cohesión de Clústeres (Silhouette)", interactive=False)
            eval_kpi_2 = gr.Textbox(label="Separación de Clústeres (Davies-Bouldin)", interactive=False)
            eval_kpi_3 = gr.Textbox(label="Varianza de Clústeres (Calinski-Harabasz)", interactive=False)
        with gr.Row():
            eval_kpi_4 = gr.Textbox(label="Confianza Promedio (Sentimiento)", interactive=False)
            eval_kpi_5 = gr.Textbox(label="Confianza Promedio (Emociones)", interactive=False)
            eval_kpi_6 = gr.Textbox(label="Compresión de Vocabulario (Limpieza)", interactive=False)
        with gr.Row():
            plot_sent_conf = gr.Plot()
            plot_emot_conf = gr.Plot()
        with gr.Row():
            plot_correlation = gr.Plot()
        with gr.Row():
            gr.Markdown("""
            ### 📚 Benchmarks de Validación Externa (Modelos SOTA)
            - **Sentiment Analysis (RoBERTa / robertuito-sentiment)**:
              - *Exactitud (Accuracy):* ~82.4% en Corpus TASS (Español).
              - *F1-Score:* ~81.8% en detección de polaridad (Positivo, Negativo, Neutro).
            - **Emotion Detection (RoBERTa / robertuito-emotion)**:
              - *Exactitud (Accuracy):* ~76.2% en detección de 7 emociones granulares.
              - *F1-Score:* ~75.4% de macro F1.
            - **POS Tagging & NER (spaCy es_core_news_sm)**:
              - *Exactitud POS:* 96.5% en categorización de categorías gramaticales.
              - *NER F1-Score:* ~89.2% en extracción de Entidades Nombradas.
            """)
    return (eval_kpi_1, eval_kpi_2, eval_kpi_3, eval_kpi_4, eval_kpi_5, eval_kpi_6), (plot_sent_conf, plot_emot_conf, plot_correlation)

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

        # ── PANTALLA 1: Carga ──
        with gr.Column(visible=True) as loading_screen:
            gr.Markdown("### ⚙️ El sistema de IA requiere inicializarse.")
            btn_start = gr.Button("🚀 Iniciar Procesamiento NLP", variant="primary", size="lg")
            loading_status = gr.Textbox(label="Estado del Servidor", value="Esperando inicio...", interactive=False)

        # ── PANTALLA 2: Dashboard ──
        with gr.Column(visible=False) as dashboard_screen:
            with gr.Tabs():
                kpi_1, kpi_2, kpi_3, kpi_4, kpi_nss, kpi_nss_w, kpi_pol, btn_export, wc_html, diagnostic_md = _build_summary_tab()
                txt_cluster, plot_clust_sil, plot_clust_tsne, plot_clust_size = _build_clustering_tab()
                plot_sent_dist, plot_sent_bar, plot_sent_emo, plot_sent_3d = _build_perception_tab()
                plot_sunburst = _build_hierarchy_tab()
                plot_hist, plot_temporal, plot_auth, plot_likes, plot_words = _build_eda_tab()
                plot_tf_top, plot_tf_bi, plot_tf_tri = _build_semantic_tab()
                plot_ner_top, plot_ner_type, plot_ner_pos = _build_ner_tab()
                rag_mode, chatbot, msg, clear = _build_rag_tab()
                eval_kpis, eval_plots = _build_evaluation_tab()
                eval_kpi_1, eval_kpi_2, eval_kpi_3, eval_kpi_4, eval_kpi_5, eval_kpi_6 = eval_kpis
                plot_sent_conf, plot_emot_conf, plot_correlation = eval_plots

        outputs = [
            loading_screen, dashboard_screen, loading_status,
            kpi_1, kpi_2, kpi_3, kpi_4, 
            kpi_nss, kpi_nss_w, kpi_pol,
            wc_html, diagnostic_md,
            plot_hist, plot_temporal, plot_auth, plot_likes, plot_words,
            plot_tf_top, plot_tf_bi, plot_tf_tri,
            plot_ner_top, plot_ner_type, plot_ner_pos,
            plot_sent_dist, plot_sent_bar, plot_sent_emo, plot_sent_3d,
            txt_cluster, plot_clust_sil, plot_clust_tsne, plot_clust_size,
            plot_sunburst,
            eval_kpi_1, eval_kpi_2, eval_kpi_3,
            eval_kpi_4, eval_kpi_5, eval_kpi_6,
            plot_sent_conf, plot_emot_conf, plot_correlation
        ]

        # ── EVENTOS ──
        msg.submit(respond, [msg, chatbot, rag_mode], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        btn_export.click(fn=export_data, inputs=[], outputs=btn_export)
        
    return app, btn_start, outputs
