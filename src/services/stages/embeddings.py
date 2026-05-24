"""
============================================================
Módulo: services/stages/embeddings.py
============================================================
Etapa de generación de embeddings multilingües.
"""

from src.services.stages.base import PipelineStage
from src.models.embeddings import compute_embeddings


class EmbeddingsStage(PipelineStage):
    def __init__(self):
        super().__init__("embeddings", "⏳ Fase 5/7: Generando vectores semánticos (Embeddings)...")

    def run(self, context: dict, progress=None) -> dict:
        context['embeddings'] = compute_embeddings(context['df']['text_clean'].tolist(), progress=progress)
        
        # Delegar el guardado de checkpoint al repositorio
        if 'repository' in context:
            context['repository'].save_checkpoint_stage5(context['embeddings'])
        return context
