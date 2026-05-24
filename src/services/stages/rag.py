"""
============================================================
Módulo: services/stages/rag.py
============================================================
Etapa de indexación y armado del motor cognitivo RAG.
"""

from src.services.stages.base import PipelineStage
from src.models.rag import RAGSystem


class RAGStage(PipelineStage):
    def __init__(self):
        super().__init__("rag", "⏳ Fase 7/7: Indexando asistente cognitivo (RAG)...")

    def run(self, context: dict, progress=None) -> dict:
        context['rag'] = RAGSystem(context['df'])
        return context
