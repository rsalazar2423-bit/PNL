"""
============================================================
Módulo: models/rag/generator.py
============================================================
Carga de modelos generativos nativos e inferencia de texto.
"""

import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class AIGenerator:
    """
    Clase dedicada a la carga de modelos Transformers e inferencia de texto generativo.
    """
    def __init__(self):
        self.generator = None
        self.model_loaded = False

    def load_model(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> bool:
        """
        Carga el modelo generativo nativo bajo demanda.
        """
        if self.model_loaded:
            return True

        print(f"\n   [RAG-IA] Carga de IA Generativa ({model_name})...")
        try:
            from transformers import pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   [RAG-IA] Hardware detectado: {device.upper()}")
            
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

    def _build_prompt(self, retrieved: list, question: str) -> str:
        """
        Construye el prompt de sistema y usuario a partir de los documentos recuperados.
        """
        context_parts = []
        for i, doc in enumerate(retrieved[:8], 1):
            sent = {'POS': 'Positivo', 'NEG': 'Negativo', 'NEU': 'Neutral'}.get(doc.get('sentiment', ''), 'Desconocido')
            context_parts.append(f"- Comentario {i} (Sentimiento: {sent}, Likes: {doc['likes']}): \"{doc['text']}\"")
        context = "\n".join(context_parts)

        return f"""<|im_start|>system
Eres un Analista de Inteligencia de Negocios y Opinión Pública experto.
Tu objetivo es analizar los comentarios provistos y responder a la pregunta del usuario.
REGLAS ESTRICTAS:
1. Responde SIEMPRE usando viñetas (bullet points) o párrafos cortos para mayor claridad.
2. Mantén un tono profesional, objective y ejecutivo.
3. Menciona el sentimiento general de los comentarios si es relevante.
4. Si los comentarios no responden la pregunta, indica que no hay suficiente información.
<|im_end|>
<|im_start|>user
CONTEXTO (Comentarios reales recuperados):
{context}

PREGUNTA: {question}

Redacta un resumen analítico basado ÚNICAMENTE en el contexto proporcionado.<|im_end|>
<|im_start|>assistant
"""

    def generate(self, retrieved: list, question: str) -> str:
        """
        Formula el prompt de contexto y genera un resumen ejecutivo.
        """
        prompt = self._build_prompt(retrieved, question)
        out = self.generator(
            prompt, 
            max_new_tokens=300, 
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
            do_sample=True,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        answer = out[0]['generated_text'].strip()
        if not answer.startswith("#") and not answer.startswith("**"):
            answer = f"### 💡 Análisis Ejecutivo\n\n{answer}"
        return answer
