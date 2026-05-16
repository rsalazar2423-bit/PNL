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

## 🔋 Eficiencia de Recursos
La arquitectura permite cargar los modelos de forma "perezosa" (Lazy Loading). Los recursos de RAM se liberan o se reutilizan entre fases del pipeline para asegurar que la aplicación pueda correr en máquinas locales estándar.
