"""
============================================================
Módulo: services/pipeline_service.py
============================================================
Servicio central de orquestación del pipeline NLP (Orquestador).

Responsabilidad Única:
    Coordinar secuencialmente la ejecución de las etapas de NLP.
"""

import gc
import os
from src.services.pipeline_cache_repository import PipelineCacheRepository
from src.services.stages import (
    CleaningStage,
    TFIDFStage,
    NERStage,
    SentimentStage,
    EmbeddingsStage,
    ClusteringStage,
    RAGStage
)


class NLPWorkflowService:
    """
    Clase de servicio que encapsula el flujo de trabajo completo del análisis.
    """

    def __init__(self, data_path: str = "comentarios_oviedo_full.csv"):
        self.data_path = data_path
        self.repository = PipelineCacheRepository(data_path)

    def run_stage_by_stage(self, progress=None):
        """
        Ejecuta el pipeline fase por fase, yield status y datos intermedios.
        
        Yields:
            tuple: (mensaje_estado, es_exito, models_data)
        """
        # 1. Comprobar e invalidar caché si el CSV original cambió
        if self.repository.invalidate_cache_if_stale():
            yield "🔄 Archivo original modificado detectado. Re-procesando dataset...", False, None

        # 2. Bypass / Caché directo
        if self.repository.has_valid_cache():
            yield "⚡ Caché detectado. Saltando procesamiento ML (Carga instantánea)...", True, None
            return

        context = {
            'data_path': self.data_path,
            'repository': self.repository,
            'df': None,
            'tfidf_data': None,
            'ner_pos_data': None,
            'embeddings': None,
            'cluster_results': None,
            'rag': None
        }

        # 3. Cargar checkpoints si existen para saltar etapas
        start_stage_idx = 0
        
        if os.path.exists("pipeline_checkpoint_stage4.pkl"):
            yield "⚡ Checkpoint de sentimientos detectado. Cargando datos previos...", False, None
            df, tfidf_data, ner_pos_data = self.repository.load_checkpoint_stage4()
            if df is not None:
                context['df'] = df
                context['tfidf_data'] = tfidf_data
                context['ner_pos_data'] = ner_pos_data
                start_stage_idx = 4  # Salta a la etapa 5 (índice 4: EmbeddingsStage)

        if os.path.exists("pipeline_checkpoint_stage5.pkl"):
            yield "⚡ Checkpoint de embeddings detectado. Cargando vectores semánticos...", False, None
            embeddings = self.repository.load_checkpoint_stage5()
            if embeddings is not None:
                context['embeddings'] = embeddings
                if start_stage_idx == 4:
                    start_stage_idx = 5  # Salta a la etapa 6 (índice 5: ClusteringStage)

        # 4. Ejecutar las etapas del pipeline de forma iterativa (Patrón Command)
        stages = [
            CleaningStage(),
            TFIDFStage(),
            NERStage(),
            SentimentStage(),
            EmbeddingsStage(),
            ClusteringStage(),
            RAGStage()
        ]

        for i in range(start_stage_idx, len(stages)):
            stage = stages[i]
            yield stage.description, False, None
            context = stage.run(context, progress=progress)

        # 5. Guardar resultados finales
        yield "💾 Guardando resultados (Parquet + Modelos)...", False, None
        self.repository.save_final_results(
            context['df'], 
            context['tfidf_data'], 
            context['ner_pos_data'], 
            context['cluster_results'], 
            context['rag']
        )
        
        models_data = {
            'tfidf': context['tfidf_data'], 
            'ner_pos': context['ner_pos_data'],
            'cluster_results': context['cluster_results'], 
            'rag': context['rag']
        }
        
        gc.collect()
        yield "✅ Procesamiento finalizado con éxito.", True, models_data
