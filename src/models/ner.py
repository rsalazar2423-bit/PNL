"""
============================================================
Módulo: models/ner.py
============================================================
Extracción de Entidades Nombradas (NER) y Categorización
Gramatical (POS Tagging) con spaCy.

Responsabilidad Única:
    Procesar los textos con spaCy para extraer entidades y
    categorías gramaticales. Retorna frecuencias consolidadas.
    NO genera gráficos.

Ejemplo de uso:
    >>> from src.models.ner import extract_entities
    >>> result = extract_entities(df)
    >>> print(result['top_entities'][:5])
"""

from collections import Counter
import spacy
import pandas as pd


def extract_entities(df: pd.DataFrame, batch_size: int = 500) -> dict:
    """
    Ejecuta el análisis NER y POS Tagging sobre el corpus.

    Procesa los textos en lotes para extraer entidades nombradas
    (personas, lugares, organizaciones) y categorías gramaticales
    (sustantivos, verbos, adjetivos).

    Args:
        df (pd.DataFrame): Dataset con la columna 'text_clean'.
        batch_size (int): Tamaño del lote para nlp.pipe(). Default: 500.

    Returns:
        dict: Diccionario con claves:
            - 'top_entities' (list[tuple]): Top 20 entidades como (entidad, freq).
            - 'entity_type_counts' (dict): Conteo por tipo de entidad {PER: n, LOC: n, ...}.
            - 'top_pos_tags' (list[tuple]): Top 10 categorías POS como (tag, freq).
    """
    print("\n" + "=" * 60)
    print("🏷️  PASO 4: Reconocimiento de Entidades (NER) y POS Tagging")
    print("=" * 60)

    try:
        print("   [NER/POS] Cargando modelo de lenguaje 'es_core_news_sm'...")
        nlp = spacy.load("es_core_news_sm", disable=["parser"])
    except OSError:
        import subprocess
        print("   [NER/POS] Modelo no encontrado. Descargando modelo desde spaCy...")
        subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
        nlp = spacy.load("es_core_news_sm", disable=["parser"])

    texts = df['text_clean'].tolist()
    all_entities = []
    all_entity_labels = []
    all_pos_tags = []

    print("   [NER/POS] Procesando textos en lotes...")
    for doc in nlp.pipe(texts, batch_size=batch_size):
        for ent in doc.ents:
            all_entities.append(f"{ent.text.lower()} [{ent.label_}]")
            all_entity_labels.append(ent.label_)
        for token in doc:
            if not token.is_stop and not token.is_punct and token.text.strip():
                all_pos_tags.append(token.pos_)

    print("   [NER/POS] Extracción finalizada. Liberando memoria...")
    del nlp

    top_entities = Counter(all_entities).most_common(20)
    entity_type_counts = dict(Counter(all_entity_labels))
    top_pos_tags = Counter(all_pos_tags).most_common(10)

    print("✅ [NER/POS] Análisis de Entidades completado.")
    return {
        'top_entities': top_entities,
        'entity_type_counts': entity_type_counts,
        'top_pos_tags': top_pos_tags,
    }
