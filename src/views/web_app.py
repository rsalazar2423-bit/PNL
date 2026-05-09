"""
============================================================
Módulo: web_app.py
============================================================
Construye la interfaz web interactiva usando Gradio Blocks.

Este módulo es responsable de:
    - Definir el layout completo del dashboard (pestañas, KPIs, gráficos).
    - Gestionar el flujo asíncrono: pantalla de carga → procesamiento → dashboard.
    - Conectar el Chatbot RAG Dual (Rústico / IA) con la interfaz.
    - Exportar el dataset procesado a CSV bajo demanda.

Dependencias:
    - gradio >= 5.0 (Blocks API, Chatbot con formato messages)
    - joblib (para leer el caché de resultados del pipeline)

Ejemplo de uso:
    >>> from src.web_app import create_web_app
    >>> app = create_web_app(pipeline_function)
    >>> app.launch(server_name="0.0.0.0", server_port=7860)
"""

import gradio as gr
import pandas as pd
import joblib
import tempfile
import traceback

# ─────────────────────────────────────────────────────────
# TEMA OSCURO NATIVO
# ─────────────────────────────────────────────────────────
dark_mode_js = """
function force_dark_mode() {
    document.body.classList.add('dark');
}
"""

# ─────────────────────────────────────────────────────────
# LÓGICA DE CALLBACKS (Controladores)
# ─────────────────────────────────────────────────────────




def load_dashboard_ui():
    """
    Carga los resultados desde la caché y mapea cada gráfico y KPI
    a su componente correspondiente de la interfaz Gradio.

    Returns:
        list: 28 valores en orden estricto correspondientes a la lista `outputs`.
    """
    try:
        results = joblib.load('pipeline_results.pkl')
        
        eda = results['eda']
        tfidf = results['tfidf']
        ner = results['ner_pos']
        sent = results['sentiment']
        clust = results['clustering']
        rag = results['rag']
        
        wc_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{eda["wordcloud_b64"]}" style="width:100%; max-width: 1000px; border-radius:8px;"></div>'
        
        return [
            gr.update(visible=False), # loading_screen
            gr.update(visible=True),  # dashboard_screen
            rag,                      # rag_state
            eda['stats']['total_comments'], eda['stats']['unique_authors'],
            eda['stats']['total_likes'], eda['stats']['avg_length'],
            wc_html,
            eda['length_hist'], eda.get('temporal_chart', None), eda['top_authors'], eda['likes_dist'], eda['top_words'],
            tfidf['top_terms_chart'], tfidf.get('bigrams_chart'), tfidf.get('trigrams_chart'),
            ner.get('top_entities_chart'), ner.get('entity_types_chart'), ner.get('pos_chart'),
            sent['distribution_chart'], sent['bars_chart'], sent['likes_sentiment_chart'],
            f"**Segmentación Óptima:** {clust['best_k']} agrupaciones temáticas.",
            clust['silhouette_chart'], clust['tsne_chart'], clust['cluster_sizes_chart'], clust.get('topics_chart'),
            clust.get('sunburst_360', None)
        ]
    except Exception as e:
        print("\n=== ERROR EN PIPELINE ===")
        traceback.print_exc()
        print("=========================\n")
        # Fallback de seguridad si falla la carga
        return [gr.update(visible=True)] + [gr.update()] * 27


def respond(message: str, chat_history: list, rag_instance, r_mode: str):
    """
    Procesa una pregunta del usuario enviándola al Motor RAG.

    Args:
        message (str): Texto introducido por el usuario.
        chat_history (list): Historial del componente Chatbot.
        rag_instance (RAGSystem): Objeto RAG cacheados en estado.
        r_mode (str): Modo seleccionado ("rustico" o "ia").

    Returns:
        tuple: (string vacío para limpiar input, nuevo historial actualizado).
    """
    if not rag_instance:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "El modelo aún no está listo. Ejecuta el pipeline primero."})
        return "", chat_history
        
    res = rag_instance.query(message, mode=r_mode)
    answer = res['answer']
    if res.get('mode') == 'ia' and res['n_sources'] > 0:
        answer += "\n\n**Fuentes Analizadas:**\n"
        for i, s in enumerate(res['sources'][:3], 1):
            answer += f"- *\"{s['text'][:80]}...\"*\n"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history


def export_data():
    """
    Lee el DataFrame final desde la caché y lo exporta a un CSV temporal.
    
    Returns:
        str | None: Ruta absoluta al archivo generado para su descarga.
    """
    try:
        results = joblib.load('pipeline_results.pkl')
        df = results['df']
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="nlp_export_")
        df.to_csv(tmp.name, index=False)
        return tmp.name
    except Exception as e:
        print(f"Error al exportar: {e}")
        return None


# ─────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA INTERFAZ (VISTA)
# ─────────────────────────────────────────────────────────

def create_web_app(pipeline_function) -> gr.Blocks:
    """
    Construye y retorna la aplicación Gradio completa.

    Separa la lógica de presentación de la lógica de eventos. Define 
    el layout, las pestañas, y conecta los componentes con los callbacks.

    Args:
        pipeline_function (callable): Función inyectada que orquesta el análisis.

    Returns:
        gr.Blocks: Aplicación interactiva lista para el .launch().
    """
    
    dark_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0B0F19",
        body_text_color="#CBD5E1",
        block_background_fill="#111827",
        block_border_color="#1E293B",
        block_label_text_color="#94A3B8",
        block_title_text_color="#F8FAFC",
        input_background_fill="#1E293B",
        button_primary_background_fill="#2563EB",
        button_secondary_background_fill="#1E293B",
    )

    with gr.Blocks(title="Panel de Inteligencia Corporativo", theme=dark_theme, js=dark_mode_js) as app:

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
                    plot_sent_3d = gr.Plot()

                with gr.Tab("Visión 360º (Jerarquía)"):
                    plot_sunburst = gr.Plot()

                with gr.Tab("Temas Latentes (Clustering)"):
                    txt_cluster = gr.Markdown("")
                    plot_clust_sil = gr.Plot()
                    plot_clust_tsne = gr.Plot()
                    plot_clust_size = gr.Plot()
                    plot_clust_top = gr.Plot()

                with gr.Tab("Asistente Cognitivo"):
                    rag_mode = gr.Radio(["rustico", "ia"], label="Modo de Búsqueda", value="rustico")
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(label="Pregunta al modelo:")
                    clear = gr.Button("Limpiar")

        # ── EVENTOS (Callbacks) ──
        msg.submit(respond, [msg, chatbot, rag_state, rag_mode], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        btn_export.click(fn=export_data, inputs=[], outputs=btn_export)

        outputs = [
            loading_screen, dashboard_screen, rag_state,
            kpi_1, kpi_2, kpi_3, kpi_4, wc_html,
            plot_hist, plot_temporal, plot_auth, plot_likes, plot_words,
            plot_tf_top, plot_tf_bi, plot_tf_tri,
            plot_ner_top, plot_ner_type, plot_ner_pos,
            plot_sent_dist, plot_sent_bar, plot_sent_3d,
            txt_cluster, plot_clust_sil, plot_clust_tsne, plot_clust_size, plot_clust_top,
            plot_sunburst
        ]
        
        def run_pipeline_inner(progress=gr.Progress()):
            """Envoltura local para mantener la firma limpia para gr.Progress()"""
            try:
                pipeline_function(progress)
                return "✅ ¡Procesamiento completado! Cargando gráficos..."
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"⚠️ Error: {str(e)}"

        # Enlazar directamente a la función anidada para que Gradio introspecte el gr.Progress()
        btn_start.click(
            fn=run_pipeline_inner, 
            inputs=[], outputs=[loading_status], show_progress="full"
        ).then(
            fn=load_dashboard_ui, inputs=[], outputs=outputs, show_progress="hidden"
        )

    return app
