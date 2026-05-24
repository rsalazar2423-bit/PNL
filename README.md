# 🧠 PNL: Advanced NLP & Audience Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-State_of_the_Art-orange.svg)
![Architecture](https://img.shields.io/badge/Architecture-Clean_SRP-success.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio_4.0-ff69b4.svg)

Plataforma de **grado de ingeniería** diseñada para extraer, procesar y visualizar inteligencia emocional y semántica a partir de grandes volúmenes de texto no estructurado (ej. comentarios de YouTube). 

A diferencia de los scripts analíticos tradicionales, este proyecto implementa un pipeline robusto utilizando modelos fundacionales **SOTA (State-of-the-Art)** de Inteligencia Artificial para el español, empaquetados en una arquitectura de software modular.

---

## 🚀 Capacidades Core

1. **Inteligencia Emocional Profunda:** Clasifica la polaridad y 7 emociones granulares utilizando modelos **RoBERTa** pre-entrenados en español para capturar sarcasmo y sutilezas contextuales.
2. **Clustering Semántico Espacial:** Agrupa comentarios en "temas latentes" usando **Embeddings Densos (MiniLM)**, algoritmos de K-Means, y reducción dimensional **t-SNE** para visualizaciones 3D interactivas.
3. **Optimización Extrema de Memoria (DuckDB + Parquet):** Las agregaciones de gráficos se realizan en disco mediante consultas SQL ultrarrápidas con DuckDB sobre archivos columnar Parquet, lo que reduce el uso de memoria RAM a mínimos históricos y permite procesar Big Data localmente.
4. **Asistente Cognitivo (RAG):** Motor de búsqueda integrado basado en el algoritmo **BM25**, operado de forma segura en el backend para evitar sobrecargas de serialización web.
5. **Extracción de Entidades (NER):** Identificación de personas, organizaciones y locaciones clave usando pipelines industriales de **SpaCy**.

---

## 🏗️ Arquitectura y Documentación

El código respeta estrictamente el **Principio de Responsabilidad Única (SRP)**, dividiendo la lógica de Machine Learning, la orquestación en disco (DuckDB) y la interfaz gráfica web (Gradio).

Para entender el diseño a fondo, por favor revisa la documentación oficial:

* 📄 [**Guía de Ingeniería y Registro de Auditoría**](DOCUMENTATION.md): Documentación exhaustiva del proyecto con detalles de arquitectura, mapeo de stack, políticas de manejo de errores y bugs corregidos en producción.
* 📄 [**Arquitectura del Sistema**](docs/ARCHITECTURE.md): Diagrama mental del flujo de datos 3-tier y la gestión de memoria RAM.
* 📄 [**Análisis Técnico Senior**](docs/ANALISIS_TECNICO_SENIOR.md): Una revisión crítica de los cuellos de botella algorítmicos, fortalezas matemáticas y la complejidad del clustering en alta dimensión.
* 📄 [**Tech Stack Detallado**](docs/TECH_STACK.md): Explicación del Qué, el Cómo y el Por qué de cada librería utilizada (Pysentimiento, Sentence-Transformers, DuckDB, Scikit-Learn, Rank-BM25).

---

## ⚙️ El Pipeline de Datos

El motor de la aplicación ejecuta secuencialmente:
1. **Extracción y Limpieza**: Normalización de texto crudo.
2. **Tokenización y Lematización**: Reducción morfológica con `es_core_news_sm`.
3. **Extracción de Entidades (NER)**: Extracción de información estructurada.
4. **Vectores Semánticos**: Generación de Embeddings densos.
5. **Inferencia Emocional**: Clasificación de sentimientos por lotes (Batch Processing).
6. **Agrupamiento**: Clustering K-Means sobre las proyecciones t-SNE.
7. **Guardado e Indexación**: Exportación a Parquet para DuckDB e indexación del motor RAG.

---

## 💻 Instalación y Uso (Entorno Local)

### 📋 Prerrequisitos de Sistema
Para ejecutar este proyecto de forma local, asegúrate de tener instalado en tu máquina:
* **Python**: Versión `3.10` o superior (probado hasta la versión `3.14`).
* **Git**: Para el control de versiones y clonado.
* **Archivo de Datos**: El archivo `comentarios_oviedo_full.csv` (1.3 MB) debe estar ubicado en la raíz del proyecto.

---

### Opción A: Instalación con UV (Recomendado — Ultra Rápido)
[**UV**](https://github.com/astral-sh/uv) es un gestor de paquetes de Python escrito en Rust que descarga e instala dependencias hasta 100 veces más rápido que pip.

#### 1. Instalar `uv` en tu sistema:
* **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows (PowerShell)**:
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```
* **Fallback (vía pip)**:
  ```bash
  pip install uv
  ```

#### 2. Clonar y configurar el proyecto:
```bash
# Clonar repositorio
git clone https://github.com/rsalazar2423-bit/PNL.git
cd PNL

# Crear venv y sincronizar todas las dependencias de pyproject.toml
uv sync

# Descargar modelo de lenguaje en español para spaCy
uv run python -m spacy download es_core_news_sm

# Iniciar la aplicación web
uv run python main.py
```

---

### Opción B: Instalación Tradicional (vía `pip` estándar)
Si no deseas instalar `uv`, puedes utilizar el flujo tradicional de Python:

```bash
# 1. Clonar el repositorio
git clone https://github.com/rsalazar2423-bit/PNL.git
cd PNL

# 2. Crear un entorno virtual
python -m venv .venv

# 3. Activar el entorno virtual
# En Linux/macOS:
source .venv/bin/activate
# En Windows (CMD):
.venv\Scripts\activate.bat
# En Windows (PowerShell):
.venv\Scripts\Activate.ps1

# 4. Actualizar pip e instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 5. Descargar modelo de lenguaje en español para spaCy
python -m spacy download es_core_news_sm

# 6. Lanzar la aplicación web
python main.py
```

La interfaz gráfica estará disponible de inmediato en tu navegador en `http://localhost:7860`.

---
*Diseñado con un enfoque obsesivo en el alto rendimiento local y código limpio.*
