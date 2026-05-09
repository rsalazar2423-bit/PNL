"""
============================================================
Orquestador Principal del Pipeline NLP
============================================================
Coordina la ejecución secuencial de los módulos de la capa
de modelos (models/) y la capa de presentación (views/).

Implementa la "Opción B": Carga la interfaz Gradio
inmediatamente y ejecuta el procesamiento con barra de progreso.

Arquitectura:
    main.py
    ├── src/models/   → Cálculos puros (sin gráficos)
    ├── src/views/    → Gráficos Plotly (sin cálculos)
    ├── src/core/     → Tema visual centralizado
    └── src/web_app.py → Interfaz Gradio
"""

import os
os.environ['GRADIO_THEME'] = 'dark'

import gc
import joblib
import gradio as gr

# ── Capa de Modelos (Lógica de Negocio) ──
from src.models.preprocessing import load_and_clean_data
from src.models.tfidf import compute_tfidf
from src.models.ner import extract_entities
from src.models.sentiment import predict_sentiment
from src.models.clustering import compute_clusters

# ── Capa de Vistas (Presentación) ──
from src.views.eda_charts import generate_eda_plots
from src.views.tfidf_charts import generate_tfidf_charts
from src.views.ner_charts import generate_ner_charts
from src.views.sentiment_charts import generate_sentiment_charts
from src.views.clustering_charts import generate_clustering_charts

from src.models.rag import RAGSystem

CSV_PATH = 'comentarios_oviedo_full.csv'
CACHE_FILE = 'pipeline_results.pkl'


def run_full_pipeline(progress=gr.Progress()):
    """
    Ejecuta todo el pipeline NLP reportando progreso a la interfaz web.

    Flujo:
        1. Preprocesamiento (models/preprocessing)
        2. EDA (views/eda_charts)
        3. TF-IDF (models/tfidf → views/tfidf_charts)
        4. NER/POS (models/ner → views/ner_charts)
        5. Sentimiento (models/sentiment → views/sentiment_charts)
        6. Clustering (models/clustering → views/clustering_charts)
        7. RAG (rag.py)

    Returns:
        dict: Resultados cacheables con gráficos y datos.
    """
    import time

    progress(0.0, desc="Buscando caché previo...")
    time.sleep(1)

    if os.path.exists(CACHE_FILE):
        progress(0.8, desc="Cargando resultados cacheados...")
        results = joblib.load(CACHE_FILE)

        if 'rag' not in results:
            progress(0.9, desc="Indexando Motor Cognitivo (RAG)...")
            rag = RAGSystem(results['df'])
            results['rag'] = rag
            joblib.dump(results, CACHE_FILE)

        progress(1.0, desc="¡Modelos listos!")
        return results

    # ── PASO 1: Preprocesamiento ──
    progress(0.05, desc="Paso 1: Limpieza de Texto y Lematización...")
    df = load_and_clean_data(CSV_PATH)
    gc.collect()

    # ── PASO 2: EDA (solo gráficos, no modelo) ──
    progress(0.20, desc="Paso 2: Análisis Exploratorio Volumétrico...")
    eda = generate_eda_plots(df)

    # ── PASO 3: TF-IDF (modelo + gráficos) ──
    progress(0.35, desc="Paso 3: Extracción de Vectores (TF-IDF)...")
    tfidf_data = compute_tfidf(df)
    tfidf = generate_tfidf_charts(tfidf_data)

    # ── PASO 4: NER/POS (modelo + gráficos) ──
    progress(0.50, desc="Paso 4: Procesamiento de Entidades (SpaCy)...")
    ner_data = extract_entities(df)
    ner_pos = generate_ner_charts(ner_data)

    # ── PASO 5: Sentimiento (modelo + gráficos) ──
    progress(0.65, desc="Paso 5: Análisis de Sentimiento (RoBERTa)...")
    df = predict_sentiment(df)
    sentiment = generate_sentiment_charts(df)

    # ── PASO 6: Clustering (modelo + gráficos) ──
    progress(0.80, desc="Paso 6: Modelado de Espacio Latente (t-SNE 3D)...")
    df, cluster_data = compute_clusters(df, tfidf_data['tfidf_matrix'], tfidf_data['feature_names'])
    clustering = generate_clustering_charts(df, cluster_data)

    # ── PASO 7: RAG ──
    progress(0.95, desc="Paso 7: Indexando Motor Cognitivo (RAG)...")
    rag = RAGSystem(df)

    results = {
        'df': df,
        'eda': eda,
        'tfidf': tfidf,
        'ner_pos': ner_pos,
        'sentiment': sentiment,
        'clustering': clustering,
        'rag': rag
    }

    progress(0.99, desc="Guardando caché de resultados...")
    joblib.dump(results, CACHE_FILE)

    progress(1.0, desc="¡Pipeline Finalizado!")
    return results


if __name__ == '__main__':
    from src.views.web_app import create_web_app
    app = create_web_app(run_full_pipeline)
    is_colab = os.getenv('COLAB_DEPLOY', '0') == '1'
    app.launch(server_name="0.0.0.0", server_port=7860, share=is_colab)
