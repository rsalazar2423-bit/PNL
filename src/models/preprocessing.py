"""
============================================================
Módulo: models/preprocessing.py
============================================================
Carga, limpieza, normalización y lematización de comentarios.

Responsabilidad Única:
    Cargar y limpiar los comentarios de entrada, y enriquecer el
    dataset con lemas y tokens utilizando spaCy y NLTK.
"""

import re
import gc
from typing import List, Tuple, Set, Union
import pandas as pd
import spacy
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords

# Diccionario de unificación léxica para normalización
LEXICAL_MAP = {
    'voto': 'votar',
    'votos': 'votar',
    'votacion': 'votar',
    'votaciones': 'votar',
    'votando': 'votar',
    'camilito': 'camilo'
}


def _ensure_nltk_data() -> None:
    """Asegura la disponibilidad local de recursos NLTK (stopwords y punkt)."""
    for resource in ['stopwords', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


def _parse_likes(likes_str: Union[str, float, int]) -> int:
    """Parsea formatos textuales de likes (ej. '1.2K', '3M') a enteros.

    Args:
        likes_str: Representación de likes (textual o numérica).

    Returns:
        Cantidad de likes entera. 0 si es nulo o inválido.
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


def _strip_accents(text: str) -> str:
    """Remueve vocales con tildes en español preservando 'ñ' y 'Ñ'.

    Args:
        text: Cadena de texto a procesar.

    Returns:
        Cadena normalizada sin vocales acentuadas.
    """
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u',
        'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u', 'Ü': 'u'
    }
    for accented, clean in replacements.items():
        text = text.replace(accented, clean)
    return text


def _clean_text(text: str) -> str:
    """Limpia URLs, menciones, marcas de tiempo y caracteres especiales.

    Args:
        text: Texto original del comentario.

    Returns:
        Texto en minúsculas sin caracteres especiales y sin espacios redundantes.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)
    text = re.sub(r'[^\w\s\u00C0-\u00FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def _get_custom_stopwords() -> Set[str]:
    """Une las stopwords en español de NLTK con términos propios del canal.

    Returns:
        Conjunto de stopwords normalizadas sin acentos.
    """
    _ensure_nltk_data()
    spanish_stops = set(stopwords.words('spanish'))
    custom_stops = {'jajaja', 'jaja', 'xd', 'like', 'video', 'si', 'no', 'ya', 'bien'}
    all_stops = spanish_stops | custom_stops
    return {_strip_accents(w) for w in all_stops}


def _normalize_token(text: str, lemma: str) -> Tuple[str, str]:
    """Aplica el mapeo de unificación léxica para normalizar el token y lema.

    Args:
        text: Token a normalizar.
        lemma: Lema a normalizar.

    Returns:
        Tupla con token y lema normalizados.
    """
    return LEXICAL_MAP.get(text, text), LEXICAL_MAP.get(lemma, lemma)


def _lemmatize_texts(
    texts: List[str], 
    nlp: spacy.language.Language, 
    batch_size: int = 1000, 
    progress_callback = None
) -> Tuple[List[List[str]], List[List[str]]]:
    """Lematiza un lote de textos filtrando stopwords y términos repetidos consecutivos.

    Args:
        texts: Lista de textos limpios.
        nlp: Instancia de spaCy configurada.
        batch_size: Tamaño del lote de procesamiento.
        progress_callback: Callback para la barra de progreso en la UI.

    Returns:
        Tupla con la lista de tokens por texto y la lista de lemas por texto.
    """
    custom_stops = _get_custom_stopwords()
    tokens_list, lemmas_list = [], []
    texts_clean = [str(t) if isinstance(t, str) else "" for t in texts]
    
    pipe_iter = nlp.pipe(texts_clean, batch_size=batch_size, disable=['ner'])
    if progress_callback:
        pipe_iter = tqdm(pipe_iter, total=len(texts_clean), desc="Lematizando comentarios")

    for doc in pipe_iter:
        doc_tokens, doc_lemmas = [], []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) >= 2:
                token_text = _strip_accents(token.text.lower())
                token_lemma = _strip_accents(token.lemma_.lower())
                
                # Unificar variantes léxicas (ej. votos -> votar)
                token_text, token_lemma = _normalize_token(token_text, token_lemma)
                
                if token_lemma not in custom_stops:
                    # Evitar duplicidad consecutiva dentro del mismo comentario (ej. "votar votar")
                    if not doc_lemmas or doc_lemmas[-1] != token_lemma:
                        doc_tokens.append(token_text)
                        doc_lemmas.append(token_lemma)
        tokens_list.append(doc_tokens)
        lemmas_list.append(doc_lemmas)
    return tokens_list, lemmas_list


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Carga y limpia la estructura inicial del dataset (I/O).

    Args:
        csv_path: Ruta al archivo CSV.

    Returns:
        DataFrame sin duplicados de comentarios y con likes numéricos.
    """
    df = pd.read_csv(csv_path)
    if 'is_reply' in df.columns:
        df = df.drop(columns=['is_reply'])
    df = df.drop_duplicates(subset='comment_id').drop_duplicates(subset='text')
    df['likes'] = df['likes'].apply(_parse_likes)
    return df


def enrich_nlp_features(df: pd.DataFrame, progress_callback = None) -> pd.DataFrame:
    """Aplica el procesamiento de spaCy y enriquece el DataFrame con tokens y lemas.

    Args:
        df: DataFrame limpio obtenido de load_raw_data.
        progress_callback: Callback para actualizar barra de progreso.

    Returns:
        DataFrame enriquecido con columnas 'tokens', 'lemmas' y 'lemmas_text'.
    """
    df['text_clean'] = df['text'].apply(_clean_text)
    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)

    print(f"\n📝 Procesando {len(df):,} comentarios con spaCy...")
    nlp = spacy.load('es_core_news_sm', disable=['parser'])
    
    df['tokens'], df['lemmas'] = _lemmatize_texts(
        texts=df['text_clean'], 
        nlp=nlp, 
        progress_callback=progress_callback
    )
    df['lemmas_text'] = df['lemmas'].apply(lambda x: ' '.join(x))
    
    del nlp
    gc.collect()
    return df


def load_and_clean_data(csv_path: str, progress_callback = None) -> pd.DataFrame:
    """Función Fachada que orquesta la carga, limpieza y procesamiento lingüístico.

    Args:
        csv_path: Ruta al archivo CSV original.
        progress_callback: Callback opcional para reportar progreso en Gradio.

    Returns:
        DataFrame preprocesado y listo para etapas posteriores.
    """
    print("\n" + "="*60 + "\nPASO 1: Carga y Limpieza\n" + "="*60)
    df_raw = load_raw_data(csv_path)
    return enrich_nlp_features(df_raw, progress_callback=progress_callback)
