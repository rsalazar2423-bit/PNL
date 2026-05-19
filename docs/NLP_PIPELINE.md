# 🧪 Pipeline de Procesamiento NLP

El flujo de datos del proyecto se divide en fases secuenciales que transforman comentarios crudos en inteligencia accionable.

## 🔄 Fases del Pipeline

### 1. Ingesta y Limpieza (`preprocessing.py`)
- **Carga**: Lee CSV/Excel.
- **Normalización**: Remueve caracteres especiales, URLs y emojis.
- **Lematización**: Convierte palabras a su raíz (ej: "corriendo" -> "correr") usando SpaCy para reducir la dispersión de datos.

### 2. Análisis de Sentimiento y Emociones (`sentiment.py`)
- **Polaridad**: Detecta si un comentario es Positivo, Negativo o Neutro.
- **Emociones**: Clasifica el texto en 7 categorías: *Ira, Alegría, Tristeza, Miedo, Sorpresa, Asco, Otros*.
- **Confianza**: Asigna un score de probabilidad a cada predicción.

### 3. Motor Semántico (`embeddings.py`)
- Transforma el texto lematizado en vectores de 384 dimensiones.
- Estos vectores representan el "significado" en un espacio matemático.

### 4. Segmentación Temática (`clustering.py`)
- **K-Means**: Agrupa los comentarios en clústers óptimos basados en sus vectores semánticos.
- **Etiquetado por Centroides**: Selecciona automáticamente los comentarios más cercanos al centro de cada grupo para nombrar el tema descubierto.
- **Visualización 3D**: Proyecta las 384 dimensiones a 3 dimensiones mediante t-SNE para navegación visual.

### 5. Asistente Cognitivo RAG (`rag.py`)
- **Indexación**: Crea un índice de búsqueda lexical (BM25) para recuperar información exacta a velocidad extrema.
- **Recuperación**: Busca los comentarios más relevantes ante una pregunta.
- **Generación**: Sintetiza una respuesta usando el contexto recuperado, citando las fuentes originales y operando en el backend de forma segura para evitar sobrecargas de red.

### 6. Almacenamiento y Consulta Analítica (`DuckDB`)
- **Parquet**: Exportación de todos los datos limpios e inferidos a un archivo binario columnar comprimido.
- **Agregaciones SQL**: Consultas directas al archivo Parquet para generar dinámicamente los gráficos Plotly de EDA y Sentimiento, evitando la sobrecarga en RAM del servidor web.

## 📈 Métricas de Calidad
- **Silhouette Score**: Utilizado para validar que los grupos temáticos sean cohesivos y estén bien separados, mostrando matemáticamente el `K` óptimo para la segmentación.
- **Emotion Score**: Mide la intensidad de la carga emocional detectada.
