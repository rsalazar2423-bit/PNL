# 🤖 Panel de Inteligencia de Opinión — Análisis NLP
### Estudio de Audiencia: Entrevista a Juan Daniel Oviedo

Este proyecto es una plataforma avanzada de **Procesamiento de Lenguaje Natural (NLP)** diseñada para analizar y sintetizar la opinión pública a partir de comentarios de YouTube. Combina técnicas de estadística clásica, aprendizaje no supervisado y modelos de lenguaje de última generación (LLMs).

![Status](https://img.shields.io/badge/Status-Ready-success)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Framework](https://img.shields.io/badge/Framework-Gradio-orange)
![AI](https://img.shields.io/badge/AI-RAG%20%7C%20Clustering-purple)

---

## 🚀 Características Principales

### 1. 🧠 Motor RAG Dual (Generación Aumentada por Recuperación)
*   **Modo Rústico**: Búsqueda léxica ultrarrápida usando BM25 para encontrar testimonios exactos.
*   **Modo IA**: Integración con **Qwen 2.5 (HuggingFace)** para generar resúmenes inteligentes y responder preguntas complejas sobre el corpus.

### 2. 🧩 Análisis de Segmentación (Clustering)
*   **K-Means Óptimo**: Selección automática del número de temas mediante el coeficiente de **Silhouette**.
*   **Visión 360º**: Reducción de dimensionalidad con **t-SNE 3D** para visualizar la estructura semántica de la conversación.
*   **Extracción de Temas**: Identificación de keywords y tópicos latentes (LDA).

### 3. 📊 Dashboard Corporativo "Dark Premium"
*   **Análisis de Percepción**: Clasificación de sentimiento (Positivo, Negativo, Neutro) usando modelos **RoBERTa**.
*   **Mapeo de Entidades (NER)**: Identificación de personajes, lugares y organizaciones mencionadas.
*   **Métricas Ejecutivas**: KPIs de interacción, nubes de palabras y distribución temporal.

---

## 🛠️ Instalación y Configuración

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para una gestión de dependencias ultra rápida.

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/pnl-inteligencia-opinion.git
cd pnl-inteligencia-opinion
```

### 2. Preparar el entorno e instalar dependencias
```bash
# Crear entorno virtual e instalar todo automáticamente
uv venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### 3. Descargar modelos de lenguaje
El sistema descargará automáticamente los modelos necesarios (spaCy y NLTK) al iniciar por primera vez, o puedes hacerlo manualmente:
```bash
python -m spacy download es_core_news_sm
```

---

## 💻 Uso

Para lanzar la interfaz interactiva de Gradio:

```bash
python main.py
```

Una vez iniciado, abre tu navegador en `http://localhost:7860`.

---

## 🏗️ Arquitectura del Proyecto

El proyecto sigue una estructura modular para facilitar su mantenimiento:

```text
PNL/
├── src/
│   ├── core/       # Sistema de diseño y constantes
│   ├── models/     # Lógica de negocio (NLP, RAG, Clustering)
│   └── views/      # Interfaz de usuario y gráficos
├── main.py         # Orquestador y punto de entrada
├── requirements.txt # Dependencias del proyecto
└── README.md       # Documentación
```

---

## 📈 Tecnologías Utilizadas

*   **Modelos de Lenguaje**: Transformers (HuggingFace), SpaCy, Pysentimiento.
*   **Procesamiento**: Pandas, Scikit-learn, NLTK.
*   **Visualización**: Plotly, WordCloud.
*   **Interfaz**: Gradio.

---
*Desarrollado para el análisis profundo de audiencias y opinión pública.*
