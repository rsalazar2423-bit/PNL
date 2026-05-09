"""
============================================================
Módulo: models/sentiment.py
============================================================
Análisis de polaridad/sentimiento con Pysentimiento (RoBERTa).

Responsabilidad Única:
    Clasificar cada comentario en Positivo, Negativo o Neutro
    y asignar un puntaje de confianza. NO genera gráficos.

Ejemplo de uso:
    >>> from src.models.sentiment import predict_sentiment
    >>> df = predict_sentiment(df)
    >>> print(df['sentiment'].value_counts())
"""

import pandas as pd
from pysentimiento import create_analyzer


def predict_sentiment(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Ejecuta el modelo pre-entrenado de Pysentimiento para extraer
    la polaridad de cada texto limpio.

    Si la columna 'sentiment' ya existe en el DataFrame, se omite
    la predicción (idempotente).

    Args:
        df (pd.DataFrame): Dataset con la columna 'text_clean'.
        batch_size (int): Tamaño de lote para el modelo. Default: 32.

    Returns:
        pd.DataFrame: DataFrame con dos columnas nuevas:
            - 'sentiment' (str): Etiqueta de polaridad ('POS', 'NEG', 'NEU').
            - 'sentiment_score' (float): Confianza del modelo (0.0 a 1.0).
    """
    print("\n" + "=" * 60)
    print("🎭 PASO 5: Análisis de Sentimiento y Polaridad")
    print("=" * 60)

    if 'sentiment' in df.columns:
        print("   [SENTIMIENTO] Datos ya procesados. Omitiendo predicción.")
        return df

    print("   [SENTIMIENTO] Cargando modelo Pysentimiento (RoBERTa español)...")
    analyzer = create_analyzer(task="sentiment", lang="es")

    print("   [SENTIMIENTO] Evaluando polaridad por lotes...")
    texts = df['text_clean'].tolist()
    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = analyzer.predict(batch_texts)
        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
        print(f"      - Lote {i // batch_size + 1}/{total_batches} procesado...", end="\r")

    print("\n   [SENTIMIENTO] Procesamiento por lotes completado.")

    df['sentiment'] = [res.output for res in results]
    df['sentiment_score'] = [res.probas[res.output] for res in results]

    print("   [SENTIMIENTO] Liberando modelo de memoria RAM/VRAM...")
    del analyzer

    print("✅ [SENTIMIENTO] Predicciones finalizadas.")
    return df
