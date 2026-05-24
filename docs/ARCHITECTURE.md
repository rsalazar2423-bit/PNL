# 🏗️ Arquitectura del Sistema

Este proyecto sigue una arquitectura modular de 3 capas + controladores, diseñada para facilitar la escalabilidad y el mantenimiento a largo plazo.

## 📐 Capas de Diseño

### 1. Capa Core (`src/core/`)
Contiene la configuración global del sistema:
- **Tematización**: Centraliza la paleta de colores corporativa y el diseño de la UI.
- **Utilidades**: Funciones transversales que no pertenecen a la lógica de negocio ni a la vista.

### 2. Capa de Modelos (`src/models/`)
Es el cerebro del sistema. Cada módulo es independiente y sigue el principio **SRP**:
- **Preprocessing**: Limpieza y lematización (SpaCy).
- **Sentiment**: Análisis de polaridad y emociones (Pysentimiento).
- **Embeddings**: Vectorización semántica (MiniLM).
- **Clustering**: Segmentación temática.
- **RAG**: Motor de respuestas contextuales.

### 3. Capa de Vistas (`src/views/`)
Responsable de la representación visual. Los módulos aquí son puramente declarativos:
- Transforman datos numéricos en objetos Plotly.
- Configuran el layout de Gradio.

### 4. Controladores (`src/controllers/`)
Actúan como mediadores. Gestionan los eventos de la interfaz (clicks, entradas de texto) y llaman a los modelos correspondientes, manteniendo la vista limpia de lógica.

## 🧠 Justificación Tecnológica

### ¿Por qué MiniLM para Embeddings?
Se eligió `paraphrase-multilingual-MiniLM-L12-v2` bajo la filosofía de **"Justo a la medida"**:
- **Peso**: Solo 80MB.
- **Rendimiento**: Ejecución ultra-rápida en CPU sin necesidad de GPU dedicada.
- **Precisión**: Soporte nativo para español con alta calidad semántica.

### ¿Por qué Pysentimiento (RoBERTa)?
Utilizamos modelos basados en RoBERTa adaptados específicamente para español, lo que garantiza una precisión mucho mayor en tareas de sentimiento y emociones que los modelos genéricos de traducción.

### ¿Por qué DuckDB & Parquet?
Para permitir que la aplicación escale a millones de comentarios sin consumir gigabytes de memoria RAM:
- **Lectura en Disco**: Los resultados del pipeline se exportan a un archivo de base de datos columnar `pipeline_data.parquet`.
- **Agregaciones SQL**: Los módulos de visualización (`eda_charts` y `sentiment_charts`) consultan el archivo Parquet directamente desde el disco duro en lugar de cargar el DataFrame completo a memoria. Solo los resultados agrupados y reducidos se transfieren a Plotly.

## 🔋 Eficiencia de Recursos y Estado de la UI
- **Persistencia Segregada**: Los datos analíticos se almacenan en `pipeline_data.parquet` mientras que los serializadores y el modelo RAG se almacenan en `pipeline_models.pkl` para evitar sobrecargas de E/S.
- **Caché de UI Pre-renderizada**: La interfaz completa de Gradio se almacena en `pipeline_ui_cache.pkl` una vez generada, lo que reduce las recargas posteriores de la página a menos de 0.1 segundos.
- **Lazy Loading**: Los modelos de IA se cargan de forma perezosa y sus recursos son liberados a través del recolector de basura (`gc.collect()`) después de cada fase del pipeline.
- **Evitación de Serialización Client-Side**: Para prevenir la caída de la UI ("pantalla negra"), el objeto pesado del motor de búsqueda (`RAGSystem`) se mantiene en el backend mediante un estado global de Python (`GLOBAL_RAG_SYSTEM`), en lugar de serializarse a través del navegador web usando `gr.State`.
- **Compatibilidad UI**: Los contenedores y layouts se configuran de forma permanentemente visible para evitar problemas en el motor de renderizado de Gradio 4 al inicializar gráficos interactivos de Plotly.
