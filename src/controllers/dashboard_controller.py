"""
============================================================
Módulo: controllers/dashboard_controller.py
============================================================
Controlador de Presentación y Eventos.

RESPONSABILIDAD: Mapear datos a visualizaciones y gestionar la UI.
Actúa como puente entre PipelineService y src/views.
"""

import gradio as gr
import joblib
import tempfile
import traceback
import pandas as pd

# Variable global para evitar serialización masiva de gr.State en el cliente
GLOBAL_RAG_SYSTEM = None

# Importaciones de Vistas (Generación de Gráficos)
from src.views.eda_charts import generate_eda_plots
from src.views.sentiment_charts import generate_sentiment_charts
from src.views.clustering_charts import generate_clustering_charts
from src.views.tfidf_charts import generate_tfidf_charts
from src.views.ner_charts import generate_ner_charts
from src.services.pipeline_service import NLPWorkflowService
from src.models.preprocessing import load_and_clean_data
from src.models.tfidf import compute_tfidf
from src.models.ner import extract_entities
from src.models.sentiment import predict_sentiment
from src.models.embeddings import compute_embeddings
from src.models.clustering import compute_clusters
from src.models.rag import RAGSystem

def _status(msg):
    """Helper: genera una lista de updates con solo el mensaje de estado cambiado."""
    return [gr.update(visible=True), gr.update(visible=False), gr.update(value=msg)] + [gr.update()] * 25

def handle_start_pipeline(progress=gr.Progress(track_tqdm=True)):
    """
    Controlador paso a paso: ejecuta cada fase y actualiza la UI entre cada una.
    """
    import os
    try:
        # ── BYPASS / CACHE ──
        if os.path.exists("pipeline_data.pkl") and os.path.exists("pipeline_models.pkl"):
            yield _status("⚡ Caché detectado. Saltando procesamiento ML (Carga instantánea)...")
            # En lugar de yield _status, el load_dashboard_ui devuelve la lista final
            for item in [load_dashboard_ui()]:
                yield item
            return

        # ── FASE 1 ──
        yield _status("⏳ Fase 1/7: Limpiando y lematizando comentarios...")
        df = load_and_clean_data("comentarios_oviedo_full.csv", progress_callback=progress)

        # ── FASE 2 ──
        yield _status("⏳ Fase 2/7: Extrayendo características TF-IDF...")
        tfidf_data = compute_tfidf(df)

        # ── FASE 3 ──
        yield _status("⏳ Fase 3/7: Detectando entidades nombradas (NER)...")
        ner_pos_data = extract_entities(df)

        # ── FASE 4 ──
        yield _status("⏳ Fase 4/7: Analizando sentimientos y emociones (RoBERTa)... ☕ Esta fase tarda ~5 min")
        df = predict_sentiment(df, progress=progress)

        # ── FASE 5 ──
        yield _status("⏳ Fase 5/7: Generando vectores semánticos (Embeddings)...")
        embeddings = compute_embeddings(df['text_clean'].tolist(), progress=progress)

        # ── FASE 6 ──
        yield _status("⏳ Fase 6/7: Agrupando por temas (Clustering)...")
        df, cluster_results = compute_clusters(df, embeddings)

        # ── FASE 7 ──
        yield _status("⏳ Fase 7/7: Indexando asistente cognitivo (RAG)...")
        rag_system = RAGSystem(df)

        # ── GUARDAR ──
        yield _status("💾 Guardando resultados (Parquet + Modelos)...")
        import gc
        
        # 1. Guardar DataFrame en Pickle y Parquet (para DuckDB)
        joblib.dump(df, "pipeline_data.pkl")
        df.to_parquet("pipeline_data.parquet", index=False)
        
        # 2. Guardar modelos y metadata pesada
        models_data = {
            'tfidf': tfidf_data, 'ner_pos': ner_pos_data,
            'cluster_results': cluster_results, 'rag': rag_system
        }
        joblib.dump(models_data, "pipeline_models.pkl")
        gc.collect()

        # ── CARGAR DASHBOARD ──
        yield load_dashboard_ui()

    except Exception as e:
        print(f"\n❌ ERROR EN CONTROLADOR: {e}")
        traceback.print_exc()
        yield _status(f"⚠️ Error en el pipeline: {str(e)}")

def load_dashboard_ui():
    """
    Carga los datos crudos, genera las visualizaciones y mapea a la UI.
    """
    try:
        import pandas as pd
        # 1. Cargar datos del caché (Pickle estándar para evitar bugs de tipos en Plotly/orjson)
        df = joblib.load("pipeline_data.pkl")
        data = joblib.load('pipeline_models.pkl')
        
        # 2. Delegar generación de visualizaciones a la capa de Vista
        print("   [CONTROLLER] Generando visualizaciones a partir de datos crudos...")
        parquet_path = "pipeline_data.parquet"
        eda = generate_eda_plots(parquet_path)
        sent = generate_sentiment_charts(parquet_path)
        clust = generate_clustering_charts(df, data['cluster_results'])
        tfidf = generate_tfidf_charts(data['tfidf'])
        ner = generate_ner_charts(data['ner_pos'])
        
        global GLOBAL_RAG_SYSTEM
        GLOBAL_RAG_SYSTEM = data['rag']
        
        # 3. Preparar componentes específicos
        wc_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{eda["wordcloud_b64"]}" style="width:100%; max-width: 1000px; border-radius:8px;"></div>'
        
        # 4. Retornar lista de componentes para Gradio (28 elementos)
        return [
            gr.update(visible=False), # loading_screen
            gr.update(visible=True),  # dashboard_screen
            gr.update(value="✅ ¡Dashboard cargado con éxito!"), # loading_status
            eda['stats']['total_comments'], eda['stats']['unique_authors'],
            eda['stats']['total_likes'], eda['stats']['avg_length'],
            wc_html,
            eda['length_hist'], eda.get('temporal_chart', None), eda['top_authors'], eda['likes_dist'], eda['top_words'],
            tfidf['top_terms_chart'], tfidf.get('bigrams_chart'), tfidf.get('trigrams_chart'),
            ner.get('top_entities_chart'), ner.get('entity_types_chart'), ner.get('pos_chart'),
            sent['distribution_chart'], sent['bars_chart'], sent.get('emotion_chart'), sent['likes_sentiment_chart'],
            f"**Segmentación Semántica:** {clust['best_k']} grupos detectados.",
            clust['silhouette_chart'], clust['tsne_chart'], clust['cluster_sizes_chart'],
            clust.get('sunburst_360', None)
        ]
    except Exception as e:
        print("\n=== ERROR EN EL CONTROLADOR DE UI ===")
        traceback.print_exc()
        return [gr.update(visible=True)] + [gr.update()] * 27

def respond(message: str, chat_history: list, r_mode: str):
    """
    Procesa una pregunta del usuario enviándola al Motor RAG global.
    """
    global GLOBAL_RAG_SYSTEM
    if not GLOBAL_RAG_SYSTEM:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "El modelo aún no está listo. Ejecuta el procesamiento primero."})
        return "", chat_history
        
    res = GLOBAL_RAG_SYSTEM.query(message, mode=r_mode)
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
    Exporta el DataFrame final a un CSV.
    """
    try:
        import pandas as pd
        import joblib
        df = joblib.load("pipeline_data.pkl")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="nlp_export_")
        df.to_csv(tmp.name, index=False, encoding='utf-8-sig')
        return tmp.name
    except Exception as e:
        print(f"Error al exportar: {e}")
        return None
