"""
============================================================
Módulo: rag.py
============================================================
Capa de compatibilidad para el Sistema RAG Dual.
Re-expone las clases del subpaquete src.models.rag.
"""

from src.models.rag.retriever import BM25Retriever
from src.models.rag.generator import AIGenerator
from src.models.rag.system import RAGSystem
