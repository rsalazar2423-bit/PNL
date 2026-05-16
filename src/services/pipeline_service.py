"""
============================================================
Módulo: services/pipeline_service.py
============================================================
Servicio central de orquestación del pipeline NLP.

RESPONSABILIDAD: Solo procesa datos e IA. No sabe nada de gráficos.
"""

import gc
import joblib
import pandas as pd

# Importaciones de Modelos (Lógica de Negocio)
from src.models.preprocessing import load_and_clean_data
from src.models.tfidf import compute_tfidf
from src.models.ner import extract_entities
from src.models.sentiment import predict_sentiment
from src.models.clustering import compute_clusters
from src.models.embeddings import compute_embeddings
from src.models.rag import RAGSystem

class NLPWorkflowService:
    """
    Clase de servicio que encapsula el flujo de trabajo completo del análisis.
    """

    def __init__(self, data_path: str = "comentarios_oviedo_full.csv"):
        self.data_path = data_path
        self.results_file = "pipeline_results_data.pkl" # Guardamos datos crudos

    def run_full_analysis(self, progress_callback=None) -> dict:
        """
        Ejecuta el procesamiento de IA y retorna datos enriquecidos.
        """
        def report(val, desc):
            if progress_callback:
                progress_callback(val, desc=desc)
            print(f"\n>>> {desc}")

        # 1. Carga y Limpieza
        report(0.10, "Fase 1: Carga y Limpieza de Datos...")
        df = load_and_clean_data(self.data_path, progress_callback=progress_callback)
        
        # 2. TF-IDF
        report(0.25, "Fase 2: Extracción de Características (TF-IDF)...")
        tfidf_data = compute_tfidf(df)
        
        # 3. NER (Entidades)
        report(0.40, "Fase 3: Extracción de Entidades y POS...")
        ner_pos_data = extract_entities(df)
        
        # 4. Sentimiento y Emociones
        report(0.60, "Fase 4: Inteligencia Emocional (RoBERTa)...")
        df = predict_sentiment(df, progress=progress_callback)
        
        # 5. Motor Semántico (Embeddings)
        report(0.75, "Fase 5: Generando Motor Semántico (Embeddings)...")
        embeddings = compute_embeddings(df['text_clean'].tolist(), progress=progress_callback)
        
        # 6. Clustering Semántico
        report(0.85, "Fase 6: Segmentación Temática (Clustering)...")
        df, cluster_results = compute_clusters(df, embeddings)
        
        # 7. RAG (Asistente)
        report(0.95, "Fase 7: Indexando Asistente Cognitivo (RAG)...")
        rag_system = RAGSystem(df)
        
        # Empaquetado final de DATOS (sin gráficos)
        raw_results = {
            'df': df,
            'tfidf': tfidf_data,
            'ner_pos': ner_pos_data,
            'cluster_results': cluster_results,
            'rag': rag_system
        }
        
        # Persistencia de datos crudos
        joblib.dump(raw_results, self.results_file)
        
        gc.collect()
        report(1.0, "✅ Procesamiento de datos finalizado.")
        return raw_results
