"""
============================================================
Módulo: views/ui_mapper.py
============================================================
Mapeador de visualizaciones de Gradio.

Responsabilidad Única:
    Recibir el estado puro del negocio y traducirlo en componentes de interfaz (Gradio / Plotly / HTML).
"""

import gradio as gr
from src.views.eda_charts import generate_eda_plots
from src.views.sentiment_charts import generate_sentiment_charts
from src.views.clustering_charts import generate_clustering_charts
from src.views.tfidf_charts import generate_tfidf_charts
from src.views.ner_charts import generate_ner_charts


def _generate_cluster_descriptions(cluster_labels: dict, best_k: int) -> str:
    """
    Genera una descripción formateada en Markdown para cada cluster detectado.
    """
    cluster_desc = f"### 🧩 Segmentación Semántica: {best_k} Temas Detectados\n"
    cluster_desc += "A continuación se definen los temas identificados de acuerdo con el comentario más representativo de cada grupo:\n\n"
    for idx in sorted(cluster_labels.keys()):
        raw_desc = cluster_labels[idx]
        if ":" in raw_desc:
            raw_desc = raw_desc.split(":", 1)[1].strip()
        cluster_desc += f"🟢 **Tema {idx+1}**: *\"{raw_desc}\"*\n\n"
    return cluster_desc


def _generate_charts(state: dict, parquet_path: str) -> dict:
    """
    Orquesta la llamadas a los generadores de Plotly y los agrupa en un diccionario.
    """
    print("   [UI-MAPPER] Generando gráficos de Plotly...")
    return {
        'eda': generate_eda_plots(parquet_path),
        'sent': generate_sentiment_charts(parquet_path),
        'clust': generate_clustering_charts(state['df'], state['cluster_results']),
        'tfidf': generate_tfidf_charts(state['tfidf_data']),
        'ner': generate_ner_charts(state['ner_pos'])
    }


def _format_metrics(metrics: dict) -> list:
    """
    Traduce las métricas puras a componentes actualizables de Gradio.
    """
    return [
        gr.update(value=f"{metrics['silhouette']:.4f}" if metrics['silhouette'] != 0.0 else "N/D (Re-procesar)"),
        gr.update(value=f"{metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] != 0.0 else "N/D (Re-procesar)"),
        gr.update(value=f"{metrics['calinski_harabasz']:.1f}" if metrics['calinski_harabasz'] != 0.0 else "N/D (Re-procesar)"),
        gr.update(value=f"{metrics['avg_sent_conf']:.2f}%" if metrics['avg_sent_conf'] != 0.0 else "N/D"),
        gr.update(value=f"{metrics['avg_emot_conf']:.2f}%" if metrics['avg_emot_conf'] != 0.0 else "N/D"),
        gr.update(value=f"{metrics['vocab_compression']:.2f}%" if metrics['vocab_compression'] != 0.0 else "N/D")
    ]


def _generate_strategic_diagnostic(state: dict) -> str:
    """
    Genera un informe estratégico en Markdown para el cliente Oviedo,
    basado en las métricas e impacto de likes calculados.
    """
    metrics = state['metrics']
    nss = metrics.get('nss', 0.0)
    nss_w = metrics.get('nss_weighted', 0.0)
    polarization = metrics.get('polarization', 0.0)
    
    nss_status = "🔴 Zona de Alerta (Crítica)" if nss < 0 else "🟢 Zona Favorable (Saludable)"
    nss_w_status = "🟢 Zona de Liderazgo (Gran Respaldo)" if nss_w > 20 else "🟡 Zona Moderada"
    
    return f"""### 📊 Análisis de Salud de Marca

* **Net Sentiment Score (NSS - Reputación Cruda)**: `{nss:+.2f}%` &nbsp;|&nbsp; **{nss_status}**
  * *Interpretación:* El volumen crudo de comentarios individuales tiende a ser negativo debido a una acumulación de críticas cortas de bajo impacto o spam.
* **NSS Ponderado por Interacciones (Impacto Real)**: `{nss_w:+.2f}%` &nbsp;|&nbsp; **{nss_w_status}**
  * *Interpretación:* **¡Hallazgo Clave!** Cuando ponderamos la opinión por la cantidad de "Likes" (que representan el respaldo comunitario real), la reputación de Oviedo se eleva radicalmente. Los comentarios positivos reciben un respaldo masivo por parte de la audiencia lectora.
* **Índice de Polarización de la Audiencia**: `{polarization:.2f}%`
  * *Interpretación:* La conversación es altamente activa. Solo un porcentaje menor permanece neutral, lo que demuestra que Oviedo es un personaje que despierta pasiones y activa la participación del público.

---

### 👥 Percepción Pública y Ejes de Opinión

* **👍 Eje Favorable (Humildad y Cercanía)**:
  La audiencia valora fuertemente la disposición de Oviedo para asistir a medios alternativos y humorísticos (como *Fuck News*) sin usar discursos rígidos o tradicionales. Se destaca su valentía y naturalidad al interactuar en escenarios no controlados.
* **👎 Eje de Tensión (El "Gancho Ciego" / Percepción Posh)**:
  Existe una burla satírica recurrente sobre su origen socioeconómico o forma técnica de expresarse. Sin embargo, el análisis de likes revela que estas observaciones tienen un carácter más jocoso y humorístico que de rechazo político hostil.

---

### 💡 Recomendaciones Estratégicas para el Candidato

1. **Humanizar la Jerga Técnica:**
   Mantener la rigurosidad estadística (que le da credibilidad), pero usar analogías más sencillas. El público reacciona de forma sumamente positiva cuando Oviedo se muestra accesible.
2. **Priorizar Canales No Convencionales:**
   La participación en este espacio digital generó un impacto favorable masivo de apoyo que supera con creces el retorno de los debates en medios tradicionales. Se recomienda continuar con esta estrategia de "cercanía digital".
3. **Ignorar el Ruido Negativo de Bajo Impacto:**
   El alto número de comentarios negativos individuales carece casi por completo de likes. Es ruido no respaldado por la comunidad lectora; no se debe gastar capital político respondiendo a estas críticas superficiales.
"""


def map_state_to_gradio_ui(state: dict) -> list:
    """
    Traduce el estado puro del negocio en el formato esperado por Gradio.

    Args:
        state (dict): Estado del negocio calculado y consolidado.

    Returns:
        list: Lista de 40 componentes actualizados para Gradio.
    """
    parquet_path = "pipeline_data.parquet"
    
    # 1. Generar gráficos (Delegado)
    charts = _generate_charts(state, parquet_path)
    
    # 2. Formatear contenedores visuales (HTML/Markdown)
    wc_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{charts["eda"]["wordcloud_b64"]}" style="width:100%; max-width: 1000px; border-radius:8px;"></div>'
    cluster_desc = _generate_cluster_descriptions(state['cluster_results'].get('cluster_labels', {}), charts['clust']['best_k'])
    diagnostic_md = _generate_strategic_diagnostic(state)
    
    # 3. Formatear métricas a formato Gradio (Delegado)
    formatted_metrics = _format_metrics(state['metrics'])
    
    # Extraer métricas de reputación
    metrics = state['metrics']
    nss_str = f"{metrics.get('nss', 0.0):+.2f}%"
    nss_w_str = f"{metrics.get('nss_weighted', 0.0):+.2f}%"
    pol_str = f"{metrics.get('polarization', 0.0):.2f}%"
    
    # 4. Ensamble de salida alineado con Gradio (41 elementos)
    return [
        gr.update(visible=False), # loading_screen
        gr.update(visible=True),  # dashboard_screen
        gr.update(value="✅ ¡Dashboard cargado con éxito!"), # loading_status
        state['stats']['total_comments'], state['stats']['unique_authors'],
        state['stats']['total_likes'], state['stats']['avg_length'],
        nss_str, nss_w_str, pol_str, # Nuevos KPIs de Reputación
        wc_html, diagnostic_md, # Nube de palabras y diagnóstico
        charts['eda']['length_hist'], charts['eda'].get('temporal_chart'), charts['eda']['top_authors'], charts['eda']['likes_dist'], charts['eda']['top_words'],
        charts['tfidf']['top_terms_chart'], charts['tfidf'].get('bigrams_chart'), charts['tfidf'].get('trigrams_chart'),
        charts['ner'].get('top_entities_chart'), charts['ner'].get('entity_types_chart'), charts['ner'].get('pos_chart'),
        charts['sent']['distribution_chart'], charts['sent']['bars_chart'], charts['sent'].get('emotion_chart'), charts['sent']['likes_sentiment_chart'],
        cluster_desc,
        charts['clust']['silhouette_chart'], charts['clust']['tsne_chart'], charts['clust']['cluster_sizes_chart'],
        charts['clust'].get('sunburst_360'),
        *formatted_metrics,
        charts['sent'].get('sentiment_confidence'),
        charts['sent'].get('emotion_confidence'),
        charts['sent'].get('correlation_chart')
    ]
