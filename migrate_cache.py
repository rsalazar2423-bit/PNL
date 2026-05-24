"""
============================================================
Módulo: migrate_cache.py
============================================================
Script de utilidad para migración de caché analítico.

RESPONSABILIDAD:
    Detectar si existe un archivo de caché monolítico antiguo 
    (pipeline_results_data.pkl) y segregar sus datos:
    1. Guardar el DataFrame procesado en formato Parquet ('pipeline_data.parquet')
       para optimizar lecturas columnares de DuckDB y Pandas.
    2. Guardar los modelos de NLP/ML entrenados en formato Joblib ('pipeline_models.pkl').
"""

import joblib
import pandas as pd
import os
import gc

def migrate_cache() -> None:
    """
    Realiza la migración del caché monolítico antiguo al esquema segregado moderno.
    
    Lee 'pipeline_results_data.pkl', extrae el DataFrame principal para escribirlo
    en formato Apache Parquet, y guarda las estructuras de modelos (TF-IDF, NER, 
    Clustering y RAG) en un archivo pickle separado.
    
    Returns:
        None
    """
    old_cache_path = 'pipeline_results_data.pkl'
    
    if not os.path.exists(old_cache_path):
        print("ℹ️ No se encontró caché monolítico antiguo para migrar.")
        return
        
    print("⏳ Cargando caché monolítico antiguo en memoria...")
    try:
        data = joblib.load(old_cache_path)
        
        df = data['df']
        
        # 1. Exportar el DataFrame a Parquet columnar
        print(f"💾 Guardando DataFrame procesado en 'pipeline_data.parquet' ({len(df)} registros)...")
        df.to_parquet('pipeline_data.parquet', index=False)
        
        # 2. Guardar los modelos analíticos en un pickle segregado
        print("💾 Guardando modelos de procesamiento NLP/ML en 'pipeline_models.pkl'...")
        models = {
            'tfidf': data.get('tfidf'),
            'ner_pos': data.get('ner_pos'),
            'cluster_results': data.get('cluster_results'),
            'rag': data.get('rag')
        }
        joblib.dump(models, 'pipeline_models.pkl')
        
        # Liberar memoria
        del data, df, models
        gc.collect()
        
        print("🎉 ¡Migración de caché analítico completada con éxito!")
    except Exception as e:
        print(f"❌ Error durante la migración de caché: {e}")

if __name__ == "__main__":
    migrate_cache()
