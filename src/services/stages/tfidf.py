"""
============================================================
Módulo: services/stages/tfidf.py
============================================================
Etapa de análisis de frecuencias léxicas TF-IDF.
"""

from src.services.stages.base import PipelineStage
from src.models.tfidf import compute_tfidf


class TFIDFStage(PipelineStage):
    def __init__(self):
        super().__init__("tfidf", "⏳ Fase 2/7: Extrayendo características TF-IDF...")

    def run(self, context: dict, progress=None) -> dict:
        context['tfidf_data'] = compute_tfidf(context['df'])
        return context
