"""
============================================================
Módulo: rag.py
============================================================
Sistema RAG Dual (Retrieval-Augmented Generation) 
Optimizado para Google Colab.

Ofrece dos modos de operación:
    1. Modo Rústico: Busca y devuelve directamente los 
       comentarios exactos usando BM25 (Estadístico, rápido, 0 RAM).
    2. Modo IA: Ejecuta un modelo de HuggingFace nativo
       (Qwen2.5-0.5b o similar) usando Transformers
       y aceleración por GPU para redactar resúmenes.
"""

from rank_bm25 import BM25Okapi
import torch
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)

class RAGSystem:
    def __init__(self, df, top_k=10):
        """
        Inicializa el sistema RAG (BM25 estático + Generador IA dinámico).

        Args:
            df (pd.DataFrame): Dataset preprocesado con la columna 'lemmas'.
            top_k (int): Número de documentos a recuperar por defecto.
        """
        print(f"\n🔍 Inicializando sistema RAG Dual...")
        self.df = df
        self.top_k = top_k
        self.generator = None
        self.model_loaded = False
        
        # Construir índice BM25 (Base de Conocimiento Rústica)
        print("   [RAG] Construyendo índice estadístico BM25...")
        tokenized_corpus = df['lemmas'].tolist()
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"   ✅ Índice BM25 construido ({len(df):,} documentos)")

    def load_ai_model(self, model_name="Qwen/Qwen2.5-0.5B-Instruct") -> bool:
        """
        Carga el modelo generativo nativo bajo demanda.
        
        Inicializa el pipeline de Transformers en fp16 si hay GPU disponible.
        Evita saturar la memoria RAM/VRAM si el usuario solo usa el modo rústico.

        Args:
            model_name (str): ID del modelo en HuggingFace.

        Returns:
            bool: True si el modelo se cargó exitosamente, False en caso de error.
        """
        if self.model_loaded:
            return True

        print(f"\n   [RAG-IA] Iniciando carga de IA Generativa ({model_name})...")
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   [RAG-IA] Hardware detectado: {device.upper()}")
            
            # Carga optimizada para Colab (fp16 en GPU)
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.generator = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto" if device == "cuda" else None,
                device=-1 if device == "cpu" else None,
                model_kwargs={"torch_dtype": dtype}
            )
            self.model_loaded = True
            print(f"   ✅ IA Generativa lista en memoria.")
            return True
        except ImportError:
            print("   ⚠️ Error: Faltan dependencias (transformers, accelerate).")
            return False
        except Exception as e:
            print(f"   ⚠️ Error cargando el modelo: {str(e)}")
            return False

    def _retrieve(self, query: str, top_k: int = None) -> list:
        """
        Busca los comentarios más similares a la consulta usando BM25.

        Args:
            query (str): Pregunta o término de búsqueda del usuario.
            top_k (int | None): Número de resultados. Usa self.top_k por defecto.

        Returns:
            list[dict]: Lista de documentos recuperados con texto, autor, likes y score.
        """
        if top_k is None:
            top_k = self.top_k

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                row = self.df.iloc[idx]
                doc = {
                    'text': row['text'],
                    'author': row.get('author', 'Anónimo'),
                    'likes': int(row.get('likes', 0)),
                    'score': float(scores[idx])
                }
                if 'sentiment' in row.index:
                    doc['sentiment'] = row['sentiment']
                results.append(doc)
        return results

    def query(self, question: str, mode: str = "rustico") -> dict:
        """
        Consulta el corpus de comentarios y genera una respuesta.

        Dependiendo del modo seleccionado, devuelve directamente los comentarios
        encontrados (modo rústico) o sintetiza una respuesta inteligente
        utilizando el modelo de lenguaje generativo (modo IA).

        Args:
            question (str): Pregunta del usuario.
            mode (str): "rustico" (búsqueda rápida) o "ia" (resumen inteligente).

        Returns:
            dict: Resultados estructurados con las claves:
                - 'answer' (str): Texto de la respuesta.
                - 'sources' (list): Documentos fuente utilizados.
                - 'n_sources' (int): Cantidad de fuentes encontradas.
                - 'question' (str): La pregunta original.
                - 'mode' (str): Modo en el que se ejecutó la consulta.
        """
        retrieved = self._retrieve(question)

        if not retrieved:
            return {
                'answer': "No encontré comentarios relevantes en el corpus para esta pregunta.",
                'sources': [],
                'n_sources': 0,
                'question': question,
                'mode': mode
            }

        if mode == "rustico":
            # MODO RÚSTICO: Retornar los comentarios tal cual
            answer = "**Resultados de la Búsqueda Directa (Modo Rústico):**\n\n"
            for i, doc in enumerate(retrieved[:5], 1):
                sent_icon = {'POS': '🟢', 'NEG': '🔴', 'NEU': '⚪'}.get(doc.get('sentiment', ''), '💬')
                answer += f"{i}. {sent_icon} **{doc['author']}** (👍 {doc['likes']}):\n> *\"{doc['text']}\"*\n\n"
            
            return {
                'answer': answer,
                'sources': retrieved,
                'n_sources': len(retrieved),
                'question': question,
                'mode': mode
            }
            
        elif mode == "ia":
            # MODO IA: Generar resumen inteligente
            if not self.model_loaded:
                success = self.load_ai_model()
                if not success:
                    return self.query(question, mode="rustico") # Fallback a rústico si falla
            
            context_parts = []
            for i, doc in enumerate(retrieved[:8], 1):
                context_parts.append(f"Comentario: {doc['text']}")
            context = "\n".join(context_parts)

            prompt = f"""<|im_start|>system
Eres un analista de opinión política colombiana. Responde basándote estrictamente en los comentarios dados. Si no hay información suficiente, dilo. Sé claro y directo.<|im_end|>
<|im_start|>user
CONTEXTO (Comentarios reales de YouTube):
{context}

PREGUNTA: {question}
Analiza los comentarios y responde la pregunta.<|im_end|>
<|im_start|>assistant
"""
            try:
                out = self.generator(
                    prompt, 
                    max_new_tokens=250, 
                    temperature=0.3,
                    return_full_text=False,
                    do_sample=True
                )
                answer = out[0]['generated_text'].strip()
            except Exception as e:
                answer = f"⚠️ Ocurrió un error en la generación de IA: {e}\n\n"
                answer += "Retrocediendo a Modo Rústico...\n\n"
                return self.query(question, mode="rustico")

            return {
                'answer': answer,
                'sources': retrieved,
                'n_sources': len(retrieved),
                'question': question,
                'mode': mode
            }
