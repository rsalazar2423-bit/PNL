"""
============================================================
Módulo: models/preprocessing.py
============================================================
Carga, limpieza, deduplicación y normalización del corpus de
comentarios de YouTube para prepararlo para análisis NLP.

Responsabilidad Única:
    Transformar el CSV crudo en un DataFrame limpio con columnas
    derivadas (text_clean, tokens, lemmas, lemmas_text).
    NO genera gráficos ni visualizaciones.

Pasos del pipeline:
    1. Carga del CSV crudo
    2. Deduplicación por comment_id y por texto
    3. Parseo de likes (convierte "2.9 K" → 2900)
    4. Limpieza de texto (URLs, menciones, emojis)
    5. Tokenización y lematización con spaCy
    6. Eliminación de stopwords (NLTK + lista custom colombiana)
    7. Generación de columnas derivadas

Dependencias:
    - pandas, re (stdlib)
    - spacy (modelo es_core_news_sm)
    - nltk (stopwords español)

Ejemplo de uso:
    >>> from src.models.preprocessing import load_and_clean_data
    >>> df = load_and_clean_data("comentarios_oviedo_full.csv")
    >>> print(df[['text', 'text_clean', 'lemmas']].head())
"""

import re
import gc
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords


def _ensure_nltk_data():
    """
    Descarga los recursos de NLTK necesarios si no están disponibles.

    Descarga:
        - stopwords: Lista de palabras vacías en múltiples idiomas.
        - punkt_tab: Tokenizador de oraciones basado en Punkt.
    """
    for resource in ['stopwords', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


def _parse_likes(likes_str):
    """
    Convierte el campo 'likes' de string a entero.

    YouTube muestra los likes con sufijos como 'K' (miles) o 'M' (millones).
    Esta función parsea esos formatos al valor numérico real.

    Args:
        likes_str: Valor del campo likes como string.
                   Ejemplos: "0", "123", "2.9 K", "1.2 M", "2.9\\u00a0K"

    Returns:
        int: Valor numérico de likes. Retorna 0 si no se puede parsear.

    Examples:
        >>> _parse_likes("2.9 K")
        2900
        >>> _parse_likes("1.2 M")
        1200000
        >>> _parse_likes("123")
        123
    """
    if pd.isna(likes_str):
        return 0

    likes_str = str(likes_str).strip().replace('\u00a0', ' ')

    multipliers = {'K': 1_000, 'M': 1_000_000}
    for suffix, mult in multipliers.items():
        if likes_str.upper().endswith(suffix):
            try:
                number = float(likes_str[:-1].strip().replace(',', '.'))
                return int(number * mult)
            except ValueError:
                return 0

    try:
        return int(float(likes_str.replace(',', '')))
    except ValueError:
        return 0


def _clean_text(text):
    """
    Limpia un comentario de YouTube eliminando ruido textual.

    Aplica las siguientes transformaciones en orden:
        1. Elimina URLs (http/https/www)
        2. Elimina menciones de YouTube (@usuario)
        3. Elimina timestamps de video (ej: 1:06:21)
        4. Elimina caracteres especiales preservando acentos y ñ
        5. Colapsa espacios múltiples
        6. Convierte a minúsculas

    Args:
        text (str): Texto crudo del comentario de YouTube.

    Returns:
        str: Texto limpio y normalizado.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)
    text = re.sub(r'[^\w\s\u00C0-\u00FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def _get_custom_stopwords():
    """
    Genera una lista extendida de stopwords para español colombiano.

    Combina las stopwords estándar de NLTK con palabras adicionales
    que son frecuentes en comentarios de YouTube pero no aportan
    valor semántico al análisis NLP.

    Returns:
        set: Conjunto de stopwords (español NLTK + YouTube + jerga común).
    """
    _ensure_nltk_data()
    spanish_stops = set(stopwords.words('spanish'))

    custom_stops = {
        'jajaja', 'jaja', 'jajajaja', 'jajajajaja', 'xd', 'xdd',
        'like', 'likes', 'suscribir', 'suscribirse', 'subscribe',
        'video', 'vídeo', 'videos', 'canal',
        'si', 'no', 'ya', 'bien', 'así', 'tan', 'ahí', 'aquí',
        'ser', 'hacer', 'decir', 'ver', 'dar', 'ir', 'querer',
        'pues', 'osea', 'bueno', 'mejor', 'solo', 'todo', 'toda',
        'puede', 'tiene', 'hace', 'dice', 'sabe', 'creo', 'parece',
    }

    return spanish_stops | custom_stops


def _lemmatize_texts(texts, nlp, batch_size=500):
    """
    Lematiza una serie de textos usando spaCy en modo batch.

    Args:
        texts (pd.Series): Serie de pandas con los textos limpios.
        nlp (spacy.Language): Modelo de spaCy cargado.
        batch_size (int): Tamaño del lote para procesamiento. Default: 500.

    Returns:
        tuple: (tokens_list, lemmas_list) — listas de listas por documento.
    """
    custom_stops = _get_custom_stopwords()
    tokens_list = []
    lemmas_list = []

    texts_clean = [str(t) if isinstance(t, str) else "" for t in texts]

    for doc in nlp.pipe(texts_clean, batch_size=batch_size, disable=['ner']):
        doc_tokens = []
        doc_lemmas = []
        for token in doc:
            if (not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and len(token.text) >= 2
                    and token.lemma_.lower() not in custom_stops):
                doc_tokens.append(token.text.lower())
                doc_lemmas.append(token.lemma_.lower())
        tokens_list.append(doc_tokens)
        lemmas_list.append(doc_lemmas)

    return tokens_list, lemmas_list


def load_and_clean_data(csv_path):
    """
    Pipeline completo de carga y limpieza del corpus de comentarios.

    Args:
        csv_path (str): Ruta al archivo CSV con los comentarios.

    Returns:
        pd.DataFrame: DataFrame limpio con columnas adicionales:
            - likes (int): Likes parseados a número
            - text_clean (str): Texto normalizado
            - tokens (list[str]): Lista de tokens por comentario
            - lemmas (list[str]): Lista de lemmas por comentario
            - lemmas_text (str): Lemmas concatenados con espacio
    """
    print("=" * 60)
    print("PASO 1: Carga y Limpieza de Datos")
    print("=" * 60)

    print(f"\n📥 Cargando datos desde: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Registros crudos: {len(df):,}")
    print(f"   Columnas: {list(df.columns)}")

    if 'is_reply' in df.columns:
        df = df.drop(columns=['is_reply'])
        print(f"   🗑️  Columna 'is_reply' eliminada (100% False)")

    n_before = len(df)
    df = df.drop_duplicates(subset='comment_id', keep='first')
    print(f"   🔄 Duplicados por ID eliminados: {n_before - len(df)}")

    n_before = len(df)
    df = df.drop_duplicates(subset='text', keep='first')
    print(f"   🔄 Duplicados por texto eliminados: {n_before - len(df)}")

    print(f"\n🔢 Parseando campo 'likes'...")
    df['likes'] = df['likes'].apply(_parse_likes)
    print(f"   Rango likes: {df['likes'].min()} - {df['likes'].max():,}")

    print(f"\n🧹 Limpiando texto...")
    df['text_clean'] = df['text'].apply(_clean_text)

    n_before = len(df)
    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)
    print(f"   Comentarios vacíos eliminados: {n_before - len(df)}")
    print(f"   Comentarios finales: {len(df):,}")

    print(f"\n📝 Tokenizando y lematizando con spaCy (es_core_news_sm)...")
    nlp = spacy.load('es_core_news_sm', disable=['parser'])
    df['tokens'], df['lemmas'] = _lemmatize_texts(df['text_clean'], nlp)
    df['lemmas_text'] = df['lemmas'].apply(lambda x: ' '.join(x))

    del nlp
    gc.collect()
    print(f"   ✅ spaCy liberado de memoria")

    print(f"\n{'=' * 60}")
    print(f"✅ Preprocesamiento completado: {len(df):,} comentarios limpios")
    print(f"{'=' * 60}\n")

    return df
