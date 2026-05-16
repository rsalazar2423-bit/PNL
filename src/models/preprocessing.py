"""
============================================================
Módulo: models/preprocessing.py
============================================================
Carga, limpieza y lematización con soporte de progreso.
"""

import re
import gc
import pandas as pd
import spacy
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords

def _ensure_nltk_data():
    for resource in ['stopwords', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource == 'stopwords' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

def _parse_likes(likes_str):
    if pd.isna(likes_str): return 0
    likes_str = str(likes_str).strip().replace('\u00a0', ' ')
    multipliers = {'K': 1_000, 'M': 1_000_000}
    for suffix, mult in multipliers.items():
        if likes_str.upper().endswith(suffix):
            try:
                number = float(likes_str[:-1].strip().replace(',', '.'))
                return int(number * mult)
            except ValueError: return 0
    try: return int(float(likes_str.replace(',', '')))
    except ValueError: return 0

def _clean_text(text):
    if not isinstance(text, str) or not text.strip(): return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)
    text = re.sub(r'[^\w\s\u00C0-\u00FF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def _get_custom_stopwords():
    _ensure_nltk_data()
    spanish_stops = set(stopwords.words('spanish'))
    custom_stops = {'jajaja', 'jaja', 'xd', 'like', 'video', 'si', 'no', 'ya', 'bien'}
    return spanish_stops | custom_stops

def _lemmatize_texts(texts, nlp, batch_size=1000, progress_callback=None):
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
                if token.lemma_.lower() not in custom_stops:
                    doc_tokens.append(token.text.lower())
                    doc_lemmas.append(token.lemma_.lower())
        tokens_list.append(doc_tokens)
        lemmas_list.append(doc_lemmas)
    return tokens_list, lemmas_list

def load_and_clean_data(csv_path, progress_callback=None):
    print("\n" + "="*60 + "\nPASO 1: Carga y Limpieza\n" + "="*60)
    df = pd.read_csv(csv_path)
    if 'is_reply' in df.columns: df = df.drop(columns=['is_reply'])
    df = df.drop_duplicates(subset='comment_id').drop_duplicates(subset='text')
    df['likes'] = df['likes'].apply(_parse_likes)
    df['text_clean'] = df['text'].apply(_clean_text)
    df = df[df['text_clean'].str.len() > 0].reset_index(drop=True)

    print(f"\n📝 Procesando {len(df):,} comentarios con spaCy...")
    nlp = spacy.load('es_core_news_sm', disable=['parser'])
    
    # Verificación explícita de argumentos
    df['tokens'], df['lemmas'] = _lemmatize_texts(
        texts=df['text_clean'], 
        nlp=nlp, 
        progress_callback=progress_callback
    )
    df['lemmas_text'] = df['lemmas'].apply(lambda x: ' '.join(x))
    
    del nlp
    gc.collect()
    return df
