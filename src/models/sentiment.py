"""
============================================================
Módulo: models/sentiment.py
============================================================
Análisis avanzado de sentimientos y emociones con Pysentimiento.

Este módulo implementa el análisis de polaridad (Pos/Neg/Neu) y la 
detección de emociones (Ira, Alegría, Miedo, etc.) utilizando 
modelos RoBERTa pre-entrenados para español.

Principios aplicados:
    - SRP (Single Responsibility Principle).
    - Funciones atómicas (Baja complejidad).
    - Documentación exhaustiva (Google Style).
"""

from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm
import os
import torch
from pysentimiento import create_analyzer

def _get_analyzer(task: str):
    """
    Inicializa y retorna un analizador de Pysentimiento.

    Args:
        task (str): La tarea a realizar ("sentiment" o "emotion").

    Returns:
        Analyzer: Instancia del analizador configurada para español.
    """
    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        cores_fisicos = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
        torch.set_num_threads(cores_fisicos)
        print(f"   [SENTIMIENTO] Ajustados hilos de PyTorch a núcleos físicos: {cores_fisicos}")
    print(f"   [SENTIMIENTO] Inicializando analizador '{task}' en dispositivo: {'GPU (0)' if device == 0 else 'CPU (-1)'}")
    return create_analyzer(task=task, lang="es", device=device)


def _run_batch_prediction(analyzer, texts: List[str], batch_size: int) -> List[Any]:
    """
    Ejecuta predicciones en lotes para optimizar el uso de recursos.

    Args:
        analyzer: El modelo cargado.
        texts (List[str]): Lista de textos a procesar.
        batch_size (int): Tamaño del lote.

    Returns:
        List: Resultados brutos del modelo.
    """
    results = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_res = analyzer.predict(batch)
        
        if isinstance(batch_res, list):
            results.extend(batch_res)
        else:
            results.append(batch_res)
            
    return results

def predict_polarity(texts: List[str], batch_size: int = 64, progress=None) -> List[Dict[str, Any]]:
    """
    Clasifica los textos en categorías de polaridad (Positivo, Negativo, Neutro).
    """
    print("   [SENTIMIENTO] Procesando Polaridad...")
    analyzer = _get_analyzer("sentiment")
    
    # Si hay objeto de progreso, lo usamos para el seguimiento fino
    if progress:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analizando Polaridad"):
            batch = texts[i : i + batch_size]
            results.extend(analyzer.predict(batch))
        raw_results = results
    else:
        raw_results = _run_batch_prediction(analyzer, texts, batch_size)
    
    # Liberar modelo de la memoria RAM/VRAM inmediatamente
    del analyzer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [
        {"label": res.output, "score": res.probas[res.output]} 
        for res in raw_results
    ]

def predict_emotions(texts: List[str], batch_size: int = 64, progress=None) -> List[Dict[str, Any]]:
    """
    Detecta emociones granulares en los textos (Ira, Alegría, etc.).
    """
    print("   [SENTIMIENTO] Procesando Emociones...")
    analyzer = _get_analyzer("emotion")

    if progress:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Analizando Emociones"):
            batch = texts[i : i + batch_size]
            results.extend(analyzer.predict(batch))
        raw_results = results
    else:
        raw_results = _run_batch_prediction(analyzer, texts, batch_size)
    
    # Liberar modelo de la memoria RAM/VRAM inmediatamente
    del analyzer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [
        {"label": res.output, "score": res.probas[res.output]} 
        for res in raw_results
    ]

def predict_sentiment(df: pd.DataFrame, batch_size: int = 64, progress=None) -> pd.DataFrame:
    """
    Orquestador principal del pipeline de sentimientos y emociones.

    Aplica SRP al delegar el procesamiento a funciones especializadas y 
    gestionar únicamente la integración con el DataFrame.

    Args:
        df (pd.DataFrame): Dataset con la columna 'text_clean'.
        batch_size (int): Tamaño de lote para el procesamiento.

    Returns:
        pd.DataFrame: DataFrame enriquecido con columnas sentiment y emotion.
    """
    print("\n" + "=" * 60)
    print("🎭 PASO 5: Inteligencia de Sentimientos y Emociones")
    print("=" * 60)

    # Evitar re-procesamiento si las columnas ya existen
    if 'sentiment' in df.columns and 'emotion' in df.columns:
        print("   [SENTIMIENTO] Datos ya enriquecidos. Omitiendo.")
        return df

    texts = df['text_clean'].tolist()

    # 1. Ejecutar Polaridad
    polarity_results = predict_polarity(texts, batch_size, progress)
    df['sentiment'] = [r['label'] for r in polarity_results]
    df['sentiment_score'] = [r['score'] for r in polarity_results]

    # 2. Ejecutar Emociones
    emotion_results = predict_emotions(texts, batch_size, progress)
    df['emotion'] = [r['label'] for r in emotion_results]
    df['emotion_score'] = [r['score'] for r in emotion_results]

    print("✅ [SENTIMIENTO] Análisis emocional completo.")
    return df
