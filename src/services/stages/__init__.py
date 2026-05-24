"""
============================================================
Paquete: services/stages
============================================================
Contiene las etapas modulares del pipeline NLP.
"""

from src.services.stages.base import PipelineStage
from src.services.stages.cleaning import CleaningStage
from src.services.stages.tfidf import TFIDFStage
from src.services.stages.ner import NERStage
from src.services.stages.sentiment import SentimentStage
from src.services.stages.embeddings import EmbeddingsStage
from src.services.stages.clustering import ClusteringStage
from src.services.stages.rag import RAGStage

__all__ = [
    'PipelineStage',
    'CleaningStage',
    'TFIDFStage',
    'NERStage',
    'SentimentStage',
    'EmbeddingsStage',
    'ClusteringStage',
    'RAGStage'
]
