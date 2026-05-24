"""
============================================================
Módulo: services/stages/ner.py
============================================================
Etapa de extracción de entidades nombradas y gramática (POS).
"""

from src.services.stages.base import PipelineStage
from src.models.ner import extract_entities


class NERStage(PipelineStage):
    def __init__(self):
        super().__init__("ner", "⏳ Fase 3/7: Detectando entidades nombradas (NER)...")

    def run(self, context: dict, progress=None) -> dict:
        context['ner_pos_data'] = extract_entities(context['df'])
        return context
