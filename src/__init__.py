"""
============================================================
NLP Pipeline — Análisis de Comentarios de YouTube
============================================================
Arquitectura Empresarial en 3 capas (SRP):

    src/
    ├── core/       → Utilidades compartidas (tema visual)
    ├── models/     → Lógica de negocio (cálculos puros, sin gráficos)
    │   ├── preprocessing.py
    │   ├── tfidf.py
    │   ├── ner.py
    │   ├── sentiment.py
    │   ├── clustering.py
    │   └── rag.py      → Motor RAG Dual (BM25 + Transformers)
    └── views/      → Presentación (gráficos Plotly, sin cálculos)
        ├── eda_charts.py
        ├── tfidf_charts.py
        ├── ner_charts.py
        ├── sentiment_charts.py
        ├── clustering_charts.py
        └── web_app.py  → Interfaz Gradio

Corpus: 7,363 comentarios del canal de YouTube de Camilo Díaz
        (entrevista a Juan Daniel Oviedo)
"""

__version__ = "2.0.0"
__author__ = "NLP Pipeline Project"
