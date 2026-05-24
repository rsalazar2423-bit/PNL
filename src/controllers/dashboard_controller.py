"""
============================================================
Módulo: controllers/dashboard_controller.py
============================================================
Controlador de Presentación y Eventos.

RESPONSABILIDAD: Mediación pura. Recibir eventos, delegar al negocio,
y devolver los resultados formateados al cliente web.
"""

import gradio as gr
import tempfile
import traceback

# Variable global para evitar serialización masiva de gr.State en el cliente
GLOBAL_RAG_SYSTEM = None

from src.services.pipeline_service import NLPWorkflowService
from src.services.dashboard_state_service import get_dashboard_state
from src.views.ui_mapper import map_state_to_gradio_ui


def _status(msg):
    """Genera una lista de actualizaciones visuales vacías con solo el mensaje de estado."""
    return [gr.update(visible=True), gr.update(visible=False), gr.update(value=msg)] + [gr.update()] * 38


def handle_start_pipeline(progress=gr.Progress(track_tqdm=True)):
    """
    Coordina el inicio del procesamiento NLP stage-by-stage.
    """
    try:
        service = NLPWorkflowService("comentarios_oviedo_full.csv")
        for status_msg, is_done, models_data in service.run_stage_by_stage(progress):
            if is_done:
                yield load_dashboard_ui()
            else:
                yield _status(status_msg)
    except Exception as e:
        print(f"\n❌ ERROR EN CONTROLADOR: {e}")
        traceback.print_exc()
        yield _status(f"⚠️ Error en el pipeline: {str(e)}")


def load_dashboard_ui():
    """
    Delegador puro: carga el estado analítico y lo pasa al mapeador visual.
    """
    try:
        print("   [CONTROLLER] Solicitando carga de estado analítico...")
        state = get_dashboard_state()
        
        global GLOBAL_RAG_SYSTEM
        GLOBAL_RAG_SYSTEM = state['rag']
        
        print("   [CONTROLLER] Traduciendo estado a interfaz gráfica...")
        return map_state_to_gradio_ui(state)
    except Exception as e:
        print("\n=== ERROR EN EL CONTROLADOR DE UI ===")
        traceback.print_exc()
        return [gr.update(visible=True)] + [gr.update()] * 40


def respond(message: str, chat_history: list, r_mode: str):
    """
    Procesa una pregunta del usuario enviándola al Motor RAG global.
    """
    global GLOBAL_RAG_SYSTEM
    if not GLOBAL_RAG_SYSTEM:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "El modelo aún no está listo. Ejecuta el procesamiento primero."})
        return "", chat_history
        
    res = GLOBAL_RAG_SYSTEM.query(message, mode=r_mode)
    answer = res['answer']
    if res.get('mode') == 'ia' and res['n_sources'] > 0:
        answer += "\n\n**Fuentes Analizadas:**\n"
        for i, s in enumerate(res['sources'][:3], 1):
            answer += f"- *\"{s['text'][:80]}...\"*\n"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    return "", chat_history


def export_data():
    """
    Exporta el DataFrame final a un CSV temporal para descarga.
    """
    try:
        import pandas as pd
        df = pd.read_parquet("pipeline_data.parquet")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="nlp_export_")
        df.to_csv(tmp.name, index=False, encoding='utf-8-sig')
        return tmp.name
    except Exception as e:
        print(f"Error al exportar: {e}")
        return None
