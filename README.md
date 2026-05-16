# 🧠 PNL: Advanced NLP & Audience Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-State_of_the_Art-orange.svg)
![Architecture](https://img.shields.io/badge/Architecture-Clean_SRP-success.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio_4.0-ff69b4.svg)

Plataforma de **grado de ingeniería** diseñada para extraer, procesar y visualizar inteligencia emocional y semántica a partir de grandes volúmenes de texto no estructurado (ej. comentarios de YouTube). 

A diferencia de los scripts analíticos tradicionales, este proyecto implementa un pipeline robusto utilizando modelos fundacionales **SOTA (State-of-the-Art)** de Inteligencia Artificial para el español, empaquetados en una arquitectura de software modular.

---

## 🚀 Capacidades Core

1. **Inteligencia Emocional Profunda:** Abandona los lexicones anticuados. Clasifica la polaridad y 7 emociones granulares utilizando modelos **RoBERTa** pre-entrenados para comprender el sarcasmo y el contexto.
2. **Clustering Semántico Espacial:** Agrupa miles de comentarios en "temas latentes" utilizando **Embeddings Densos (MiniLM)**, algoritmos de K-Means, y reducción dimensional **t-SNE** para visualizaciones en 3D reales.
3. **Asistente Cognitivo (RAG):** Motor de búsqueda integrado basado en el algoritmo **BM25**, que permite "chatear" con el corpus de datos para recuperar información exacta a velocidad extrema.
4. **Extracción de Entidades (NER):** Identificación automática de personas, organizaciones y locaciones clave mediante modelos industriales de **SpaCy**.

---

## 🏗️ Arquitectura y Documentación

El código respeta estrictamente el **Principio de Responsabilidad Única (SRP)**, dividiendo la lógica pesada de Machine Learning, la orquestación de datos y la interfaz gráfica web.

Para entender el diseño a fondo, por favor revisa la documentación oficial:

* 📄 [**Análisis Técnico Senior**](docs/ANALISIS_TECNICO_SENIOR.md): Una revisión crítica de los cuellos de botella algorítmicos, fortalezas matemáticas y la complejidad del clustering en alta dimensión.
* 🛠️ [**Tech Stack Detallado**](docs/TECH_STACK.md): Explicación del Qué, el Cómo y el Por qué de cada librería utilizada (Pysentimiento, Sentence-Transformers, Scikit-Learn, Rank-BM25).

---

## ⚙️ El Pipeline de Datos

El motor de la aplicación ejecuta secuencialmente:
1. **Extracción y Limpieza**: Normalización de texto crudo.
2. **Tokenización y Lematización**: Reducción morfológica con `es_core_news_sm`.
3. **Extracción de Entidades (NER)**: Extracción de información estructurada.
4. **Vectores Semánticos**: Generación de Embeddings densos.
5. **Inferencia Emocional**: Clasificación de sentimientos por lotes (Batch Processing).
6. **Agrupamiento**: Clustering K-Means sobre las proyecciones t-SNE.

---

## 💻 Instalación y Uso (Entorno Local)

Este proyecto está optimizado con **UV**, el gestor de paquetes ultrarrápido en Rust, para garantizar instalaciones deterministas.

```bash
# 1. Clonar el repositorio
git clone https://github.com/rsalazar2423-bit/PNL.git
cd PNL

# 2. Sincronizar dependencias a la velocidad de la luz
uv sync

# 3. Instalar el modelo de lenguaje base de SpaCy
uv run python -m spacy download es_core_news_sm

# 4. Lanzar la aplicación web
uv run python main.py
```

La interfaz gráfica estará disponible instantáneamente en `http://127.0.0.1:7860`.

---
*Diseñado con un enfoque obsesivo en el alto rendimiento local y código limpio.*
