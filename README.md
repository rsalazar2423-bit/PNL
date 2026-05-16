# 📊 NLP Dashboard: Análisis de Audiencia YouTube

Plataforma profesional de análisis de lenguaje natural (NLP) diseñada para extraer inteligencia emocional y temática de grandes volúmenes de comentarios de YouTube. Construida bajo una arquitectura de 3 capas y optimizada para ejecución eficiente en entornos locales.

## 🚀 Características Principales

- **Inteligencia Emocional**: Clasificación de 7 emociones granulares y polaridad usando RoBERTa.
- **Clustering Semántico**: Agrupamiento de temas por significado profundo mediante Embeddings (MiniLM).
- **Asistente Cognitivo (RAG)**: Chatbot integrado que responde preguntas basadas en el corpus de comentarios.
- **Visualización Avanzada**: Mapas semánticos 3D, visión 360º (Sunburst) y análisis temporal.
- **Arquitectura Senior**: Implementación modular siguiendo el Principio de Responsabilidad Única (SRP).

## 🛠️ Instalación y Configuración

Este proyecto utiliza **UV** como gestor de paquetes de última generación para garantizar velocidad y reproducibilidad.

1. **Clonar el repositorio**:
   ```bash
   git clone <url-del-repositorio>
   cd PNL
   ```

2. **Sincronizar el entorno**:
   ```bash
   uv sync
   ```

3. **Descargar modelos de SpaCy**:
   ```bash
   uv run python -m spacy download es_core_news_sm
   ```

## 💻 Uso

Para iniciar la aplicación en modo desarrollo:

```bash
uv run python main.py
```

La interfaz de Gradio estará disponible en: `http://127.0.0.1:7860`

## 🏗️ Estructura del Proyecto

```text
├── src/
│   ├── core/         # Configuración, temas y utilidades base
│   ├── models/       # Inteligencia (NLP, Clustering, Sentimiento)
│   ├── views/        # Interfaz de usuario y gráficos Plotly
│   └── controllers/  # Orquestación entre Vista y Modelo
├── docs/             # Documentación detallada
└── main.py           # Punto de entrada y pipeline
```

---
*Desarrollado con enfoque en alto rendimiento y arquitectura limpia.*
