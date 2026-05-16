"""
============================================================
Módulo: models/embeddings.py
============================================================
Generación de representaciones vectoriales (Embeddings) de texto.

Este módulo utiliza modelos pre-entrenados de Sentence-Transformers
para convertir texto limpio en vectores semánticos de alta dimensión,
lo que permite realizar clustering y búsquedas basadas en significado.

Principios aplicados:
    - SRP: El módulo solo se encarga de la vectorización.
    - Lazy Loading: El modelo se carga solo cuando es necesario.
    - Eficiencia: Optimizado para ejecución en CPU.
"""

from typing import List
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Modelo recomendado por su balance entre peso (80MB) y calidad semántica en español.
DEFAULT_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

def get_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Carga e inicializa el modelo de Sentence Transformers.

    Args:
        model_name (str): Identificador del modelo en HuggingFace.

    Returns:
        SentenceTransformer: Instancia del modelo cargado.
    """
    print(f"   [EMBEDDINGS] Cargando motor semántico ({model_name})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

def compute_embeddings(texts: List[str], batch_size: int = 128, progress=None) -> np.ndarray:
    """
    Transforma una lista de textos en una matriz de vectores (embeddings).
    """
    if not texts:
        return np.array([])

    model = get_embedding_model()
    
    print(f"   [EMBEDDINGS] Generando vectores para {len(texts):,} comentarios...")
    
    if progress:
        # Usamos tqdm estándar que será capturado por Gradio (track_tqdm=True)
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generando Vectores Semánticos"):
            batch = texts[i : i + batch_size]
            batch_emb = model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_emb)
        embeddings = np.vstack(embeddings)
    else:
        embeddings = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
    
    print(f"   ✅ Vectores generados exitosamente (Dimensión: {embeddings.shape[1]})")
    return embeddings
