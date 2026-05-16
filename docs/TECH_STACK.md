# 🛠️ Tech Stack: Arquitectura y Tecnologías

Este documento detalla el stack tecnológico del proyecto de Análisis NLP de Audiencias. Explica **qué** tecnologías se implementaron, **cómo** se están usando dentro del código, y **por qué** fueron la mejor elección técnica.

---

## 1. Interfaz de Usuario y Visualización

### **Gradio (`gradio`)**
*   **Qué es:** Un framework de Python para construir interfaces de usuario web (UI) para modelos de Machine Learning rápidamente.
*   **Cómo lo usamos:** Es el motor principal del frontend. Todo el dashboard, las pestañas, botones y el chatbot interactivo están construidos en `views/web_app.py` utilizando los bloques de Gradio (`gr.Blocks`).
*   **Por qué lo elegimos:** Permite desarrollar una aplicación web funcional sin necesidad de escribir HTML, CSS o JavaScript (React/Vue). Es el estándar de la industria para demostraciones rápidas de modelos de IA.

### **Plotly (`plotly`)**
*   **Qué es:** Librería de graficación interactiva.
*   **Cómo lo usamos:** Genera gráficos interactivos como el Sunburst 360º de emociones, mapas 3D de clústeres semánticos y barras apiladas (`views/clustering_charts.py` y `sentiment_charts.py`).
*   **Por qué lo elegimos:** Supera a librerías estáticas (como Matplotlib) al permitir a los usuarios hacer zoom, rotar vistas 3D y ver "tooltips" (detalles al pasar el mouse) directamente en el navegador.

---

## 2. Inteligencia Artificial y Deep Learning

### **Pysentimiento (`pysentimiento`)**
*   **Qué es:** Librería open-source basada en Transformers diseñada específicamente para el análisis de texto en español.
*   **Cómo lo usamos:** En `models/sentiment.py`, se utiliza para clasificar la **polaridad** (Positivo/Negativo) y las **emociones** (Alegría, Ira, Tristeza, etc.) de cada comentario de YouTube.
*   **Por qué lo elegimos:** Usa el modelo **RoBERTa pre-entrenado en español**. Es infinitamente superior a métodos antiguos (como diccionarios de palabras positivas/negativas) porque entiende el sarcasmo y el contexto de las frases.

### **Sentence-Transformers (`sentence-transformers`)**
*   **Qué es:** Framework de HuggingFace para computar representaciones vectoriales densas (embeddings) de oraciones y textos.
*   **Cómo lo usamos:** Transforma los textos limpios en vectores matemáticos de alta dimensionalidad (Fase de Embeddings) usando el modelo `paraphrase-multilingual-MiniLM-L12-v2`.
*   **Por qué lo elegimos:** Al entender el significado semántico profundo, permite agrupar (clusterizar) comentarios que hablan de lo mismo aunque usen palabras totalmente distintas.

---

## 3. Procesamiento Clásico de Lenguaje Natural (NLP)

### **SpaCy (`spacy` + `es_core_news_sm`)**
*   **Qué es:** Librería de NLP de grado industrial.
*   **Cómo lo usamos:** En `models/preprocessing.py` y `models/ner.py`, se usa para la **Lematización** (reducir palabras a su raíz, ej: "corriendo" -> "correr"), remover "stopwords" (y, o, el, la) y extraer Entidades Nombradas (NER: Personas, Organizaciones, Lugares).
*   **Por qué lo elegimos:** Es la librería más rápida para procesamiento de texto en producción. Su modelo pre-entrenado en español (`es_core_news_sm`) es muy ligero y exacto.

### **NLTK (`nltk`)**
*   **Qué es:** Natural Language Toolkit, una de las librerías más antiguas de NLP.
*   **Cómo lo usamos:** Lo usamos puntualmente para complementar a SpaCy en la tokenización básica y eliminación de stop-words en la etapa temprana del preprocesamiento.
*   **Por qué lo elegimos:** Su ecosistema es enorme y actúa como un respaldo sólido (fallback) para tareas sencillas de limpieza de texto.

---

## 4. Machine Learning Algorítmico y Datos

### **Scikit-Learn (`scikit-learn`)**
*   **Qué es:** La librería estándar de Machine Learning clásico en Python.
*   **Cómo lo usamos:** En `models/clustering.py` y `models/tfidf.py`, impulsa los algoritmos pesados: 
    *   **K-Means:** Para agrupar los embeddings en clústeres temáticos.
    *   **t-SNE:** Para reducir vectores de alta dimensionalidad a coordenadas X, Y, Z para el gráfico 3D.
    *   **TF-IDF Vectorizer:** Para extraer las palabras clave más frecuentes.
*   **Por qué lo elegimos:** Es extremadamente robusto, ampliamente documentado y matemáticamente confiable para aprendizaje no supervisado.

### **Pandas (`pandas`) y NumPy (`numpy`)**
*   **Qué es:** Librerías fundamentales para manipulación de matrices numéricas y bases de datos estructuradas en memoria.
*   **Cómo lo usamos:** Manejan todo el flujo de datos. El pipeline entero (`main.py`) opera sobre DataFrames de Pandas, permitiendo agregar columnas de predicciones y filtrado rápido.
*   **Por qué lo elegimos:** Son el estándar absoluto en Data Science. Su código subyacente en C asegura la velocidad máxima que Python puede ofrecer para mover datos.

---

## 5. Buscador y Recuperación de Información (RAG)

### **Rank-BM25 (`rank-bm25`)**
*   **Qué es:** Implementación del algoritmo BM25 (Okapi), el estándar de oro en recuperación de información lexical.
*   **Cómo lo usamos:** Es el núcleo del asistente cognitivo (Chatbot) en `models/rag.py`. Escanea todo el corpus tokenizado para encontrar rápidamente qué comentarios responden a la pregunta del usuario.
*   **Por qué lo elegimos:** A diferencia de las búsquedas semánticas puras (que son pesadas), BM25 es increíblemente rápido y exacto para buscar términos específicos (como "precios", "calidad", "errores").

---

## 6. Infraestructura y Entorno

### **UV (Gestor de Paquetes)**
*   **Qué es:** Un instalador y gestor de paquetes de Python escrito en Rust.
*   **Cómo lo usamos:** Reemplaza a `pip` y a `virtualenv`. Define las dependencias exactas en `pyproject.toml` y genera el entorno virtual.
*   **Por qué lo elegimos:** Es hasta 100 veces más rápido que pip. Garantiza instalaciones deterministas (con `uv.lock`) solucionando el clásico problema de "en mi máquina sí funciona".
