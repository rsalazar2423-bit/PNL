"""
============================================================
Módulo: models/rag/system.py
============================================================
Coordinador del sistema RAG (Rústico / IA).
"""

from typing import Dict, Any
import pandas as pd

from src.models.rag.retriever import BM25Retriever
from src.models.rag.generator import AIGenerator


class RAGSystem:
    """
    Fachada del sistema RAG que coordina al BM25Retriever y al AIGenerator.
    """
    def __init__(self, df: pd.DataFrame, top_k: int = 10):
        self.retriever = BM25Retriever(df, top_k)
        self.generator = AIGenerator()

    def _format_rustic_response(self, question: str, retrieved: list) -> Dict[str, Any]:
        """
        Formatea la respuesta en formato de lista directa de coincidencias.
        """
        answer = "### 🔍 Resultados de la Búsqueda (Directa)\n\n"
        answer += "*He encontrado los siguientes comentarios relevantes en la base de datos:*\n\n"
        for i, doc in enumerate(retrieved[:5], 1):
            sent_icon = {'POS': '🟢 Positivo', 'NEG': '🔴 Negativo', 'NEU': '⚪ Neutral'}.get(doc.get('sentiment', ''), '💬')
            answer += f"**{i}. {doc['author']}** &nbsp;|&nbsp; {sent_icon} &nbsp;|&nbsp; 👍 {doc['likes']} likes\n"
            answer += f"> *\"{doc['text']}\"*\n\n"
            answer += "---\n\n"
        
        return {
            'answer': answer,
            'sources': retrieved,
            'n_sources': len(retrieved),
            'question': question,
            'mode': 'rustico'
        }

    def _generate_ai_response(self, question: str, retrieved: list) -> Dict[str, Any]:
        """
        Genera y formatea la respuesta sintetizada por la IA.
        """
        if not self.generator.model_loaded:
            success = self.generator.load_model()
            if not success:
                return self._format_rustic_response(question, retrieved)
        
        try:
            answer = self.generator.generate(retrieved, question)
        except Exception as e:
            answer = f"⚠️ Ocurrió un error en la generación de IA: {e}\n\nRetrocediendo a Modo Rústico...\n\n"
            return self._format_rustic_response(question, retrieved)

        return {
            'answer': answer,
            'sources': retrieved,
            'n_sources': len(retrieved),
            'question': question,
            'mode': 'ia'
        }

    def query(self, question: str, mode: str = "rustico") -> dict:
        """
        Coordina la consulta según el modo y retorna la respuesta.
        """
        retrieved = self.retriever.retrieve(question)

        if not retrieved:
            return {
                'answer': "No encontré comentarios relevantes en el corpus para esta pregunta.",
                'sources': [],
                'n_sources': 0,
                'question': question,
                'mode': mode
            }

        if mode == "rustico":
            return self._format_rustic_response(question, retrieved)
            
        # Modo IA
        return self._generate_ai_response(question, retrieved)
