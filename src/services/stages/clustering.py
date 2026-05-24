"""
============================================================
Módulo: services/stages/clustering.py
============================================================
Etapa de segmentación por tópicos con K-Means y métricas.
"""

from src.services.stages.base import PipelineStage
from src.models.clustering import compute_clusters


class ClusteringStage(PipelineStage):
    def __init__(self):
        super().__init__("clustering", "⏳ Fase 6/7: Agrupando por temas (Clustering)...")

    def run(self, context: dict, progress=None) -> dict:
        context['df'], context['cluster_results'] = compute_clusters(context['df'], context['embeddings'])
        return context
