"""
============================================================
Módulo: models/rag/retriever.py
============================================================
Indexación y recuperación estática mediante BM25.
"""

from rank_bm25 import BM25Okapi
import pandas as pd


class BM25Retriever:
    """
    Clase dedicada a la indexación y recuperación estática mediante BM25.
    """
    def __init__(self, df: pd.DataFrame, top_k: int = 10):
        self.df = df
        self.top_k = top_k
        print("   [BM25] Construyendo índice estadístico BM25...")
        tokenized_corpus = df['lemmas'].tolist()
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"   ✅ Índice BM25 construido ({len(df):,} documentos)")

    def _format_document(self, idx: int, score: float) -> dict:
        """
        Formatea un renglón del DataFrame en un diccionario de documento RAG.
        """
        row = self.df.iloc[idx]
        doc = {
            'text': row['text'],
            'author': row.get('author', 'Anónimo'),
            'likes': int(row.get('likes', 0)),
            'score': float(score)
        }
        if 'sentiment' in row.index:
            doc['sentiment'] = row['sentiment']
        return doc

    def _lemmatize_query(self, query: str) -> list:
        """
        Lematiza y normaliza la consulta usando spaCy y el LEXICAL_MAP de preprocesamiento.
        """
        if not hasattr(self, '_nlp'):
            import spacy
            try:
                self._nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
            except OSError:
                import sys
                import subprocess
                print("   [BM25] Descargando modelo spaCy 'es_core_news_sm' para lematización de consulta...")
                subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
                try:
                    self._nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
                except Exception:
                    self._nlp = None

        if self._nlp is None:
            # Fallback a limpieza básica si spaCy no se puede cargar
            import re
            clean_query = re.sub(r'[^\w\s\u00C0-\u00FF]', ' ', query.lower())
            replacements = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u'}
            for accented, clean in replacements.items():
                clean_query = clean_query.replace(accented, clean)
            return [t for t in clean_query.split() if t]

        doc = self._nlp(query)
        query_tokens = []
        from src.models.preprocessing import LEXICAL_MAP
        
        # Reemplazos de acentos
        replacements = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u'}
        
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) >= 2:
                lemma = token.lemma_.lower()
                for accented, clean in replacements.items():
                    lemma = lemma.replace(accented, clean)
                # Unificación léxica
                lemma = LEXICAL_MAP.get(lemma, lemma)
                query_tokens.append(lemma)
        return query_tokens

    def retrieve(self, query: str, top_k: int = None) -> list:
        """
        Retorna los documentos más relevantes para la consulta.
        """
        if top_k is None:
            top_k = self.top_k

        query_tokens = self._lemmatize_query(query)
        
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self._format_document(idx, scores[idx]))
        return results
