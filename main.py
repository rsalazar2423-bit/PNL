"""
============================================================
Punto de Entrada: main.py
============================================================
Interfaz de Usuario y Lanzador del Pipeline NLP.

Este archivo es puramente DECLARATIVO. Su única responsabilidad
es configurar la interfaz de Gradio y conectar los eventos
con el PipelineService.

Arquitectura: 3 Capas + Servicios.
"""

import gradio as gr
import sys
import io

# Forzar UTF-8 en la consola para evitar errores de emojis en Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.controllers.dashboard_controller import handle_start_pipeline, respond, export_data
from src.views.web_app import create_web_app

def main():
    """
    Inicializa la aplicación Gradio.
    """
    print("\n" + "=" * 60)
    print("🚀 INICIANDO PANEL DE INTELIGENCIA NLP")
    print("=" * 60)
    
    # Construir la interfaz (Capa de Vista)
    demo, btn_start, outputs = create_web_app()
    
    # Vinculación directa de eventos (Dentro del contexto de Gradio)
    with demo:
        btn_start.click(
            fn=handle_start_pipeline,
            inputs=[],
            outputs=outputs
        )
    
    # Lanzar servidor
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

if __name__ == "__main__":
    main()
