"""
============================================================
Módulo: services/pipeline_cache_repository.py
============================================================
Administra la persistencia física del pipeline, caché y checkpoints.

Responsabilidad Única:
    Gestionar el ciclo de vida de los archivos físicos en disco.
"""

import os
import joblib
import pandas as pd


class PipelineCacheRepository:
    def __init__(self, data_path: str = "comentarios_oviedo_full.csv"):
        self.data_path = data_path
        self.cache_files = [
            "pipeline_data.parquet", 
            "pipeline_models.pkl", 
            "pipeline_checkpoint_stage4.pkl", 
            "pipeline_checkpoint_stage5.pkl"
        ]

    def invalidate_cache_if_stale(self) -> bool:
        """
        Invalida y elimina la caché física si el archivo CSV de origen es más nuevo.
        
        Returns:
            bool: True si la caché fue invalidada, False en caso contrario.
        """
        csv_mtime = os.path.getmtime(self.data_path) if os.path.exists(self.data_path) else 0
        cache_invalidated = False
        
        for cache_file in self.cache_files:
            if os.path.exists(cache_file):
                if csv_mtime > os.path.getmtime(cache_file):
                    print(f"   [REPOSITORY] Archivo original cambiado. Invalidando caché: {cache_file}")
                    try:
                        os.remove(cache_file)
                        cache_invalidated = True
                    except Exception as e:
                        print(f"   [REPOSITORY] Error al remover caché {cache_file}: {e}")
        return cache_invalidated

    def has_valid_cache(self) -> bool:
        """
        Verifica si existen los archivos de caché definitivos del pipeline.
        """
        return os.path.exists("pipeline_data.parquet") and os.path.exists("pipeline_models.pkl")

    def load_checkpoint_stage4(self) -> tuple:
        """
        Carga el checkpoint de la etapa 4 si existe.
        
        Returns:
            tuple: (df, tfidf_data, ner_pos_data) o (None, None, None)
        """
        if os.path.exists("pipeline_checkpoint_stage4.pkl"):
            try:
                ckpt = joblib.load("pipeline_checkpoint_stage4.pkl")
                print("   [REPOSITORY] Cargado checkpoint Stage 4 (sentimientos).")
                return ckpt['df'], ckpt['tfidf_data'], ckpt['ner_pos_data']
            except Exception as e:
                print(f"   [REPOSITORY] Error al cargar checkpoint Stage 4: {e}. Se reiniciará desde cero.")
        return None, None, None

    def load_checkpoint_stage5(self):
        """
        Carga el checkpoint de la etapa 5 si existe.
        
        Returns:
            np.ndarray o None
        """
        if os.path.exists("pipeline_checkpoint_stage5.pkl"):
            try:
                ckpt5 = joblib.load("pipeline_checkpoint_stage5.pkl")
                print("   [REPOSITORY] Cargado checkpoint Stage 5 (embeddings).")
                return ckpt5['embeddings']
            except Exception as e:
                print(f"   [REPOSITORY] Error al cargar checkpoint Stage 5: {e}. Se recalcularán.")
        return None

    def save_checkpoint_stage4(self, df, tfidf_data, ner_pos_data) -> None:
        """
        Guarda el checkpoint de la etapa 4 en disco.
        """
        try:
            joblib.dump({
                'df': df,
                'tfidf_data': tfidf_data,
                'ner_pos_data': ner_pos_data
            }, "pipeline_checkpoint_stage4.pkl")
            print("   [REPOSITORY] Guardado checkpoint Stage 4.")
        except Exception as e:
            print(f"   [REPOSITORY] No se pudo guardar checkpoint Stage 4: {e}")

    def save_checkpoint_stage5(self, embeddings) -> None:
        """
        Guarda el checkpoint de la etapa 5 en disco.
        """
        try:
            joblib.dump({'embeddings': embeddings}, "pipeline_checkpoint_stage5.pkl")
            print("   [REPOSITORY] Guardado checkpoint Stage 5.")
        except Exception as e:
            print(f"   [REPOSITORY] No se pudo guardar checkpoint Stage 5: {e}")

    def save_final_results(self, df: pd.DataFrame, tfidf_data: dict, ner_pos_data: dict, cluster_results: dict, rag_system) -> None:
        """
        Exporta y guarda los resultados consolidados finales y limpia checkpoints.
        """
        try:
            df.to_parquet("pipeline_data.parquet", index=False)
            
            models_data = {
                'tfidf': tfidf_data, 
                'ner_pos': ner_pos_data,
                'cluster_results': cluster_results, 
                'rag': rag_system
            }
            joblib.dump(models_data, "pipeline_models.pkl")
            print("   [REPOSITORY] Resultados finales guardados (Parquet + Modelos).")
        except Exception as e:
            print(f"   [REPOSITORY] Error al guardar resultados finales: {e}")

        # Limpiar checkpoints temporales
        for f in ["pipeline_checkpoint_stage4.pkl", "pipeline_checkpoint_stage5.pkl"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"   [REPOSITORY] Eliminado checkpoint temporal {f}.")
                except Exception as e:
                    print(f"   [REPOSITORY] No se pudo eliminar checkpoint {f}: {e}")
