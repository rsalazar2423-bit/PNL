"""
============================================================
Módulo: services/stages/sentiment.py
============================================================
Etapa de clasificación de sentimientos y emociones con RoBERTa.
"""

from src.services.stages.base import PipelineStage
from src.models.sentiment import predict_sentiment


class SentimentStage(PipelineStage):
    def __init__(self):
        super().__init__("sentiment", "⏳ Fase 4/7: Analizando sentimientos y emociones (RoBERTa)... ☕ Esta fase tarda ~5 min")

    def run(self, context: dict, progress=None) -> dict:
        context['df'] = predict_sentiment(context['df'], progress=progress)
        
        # Delegar el guardado de checkpoint al repositorio
        if 'repository' in context:
            context['repository'].save_checkpoint_stage4(
                context['df'],
                context['tfidf_data'],
                context['ner_pos_data']
            )
        return context
