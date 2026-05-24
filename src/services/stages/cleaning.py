"""
============================================================
Módulo: services/stages/cleaning.py
============================================================
Etapa de carga y limpieza del dataset.
"""

from src.services.stages.base import PipelineStage
from src.models.preprocessing import load_and_clean_data


class CleaningStage(PipelineStage):
    def __init__(self):
        super().__init__("cleaning", "⏳ Fase 1/7: Limpiando y lematizando comentarios...")

    def run(self, context: dict, progress=None) -> dict:
        context['df'] = load_and_clean_data(context['data_path'], progress_callback=progress)
        return context
