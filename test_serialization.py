import sys
import json
from src.controllers.dashboard_controller import load_dashboard_ui
import gradio as gr

def test_serialization():
    try:
        outputs = load_dashboard_ui()
        print(f"✅ Dashboard cargado. {len(outputs)} elementos.")
        
        # Iterar sobre las salidas y tratar de serializarlas (simulando lo que hace Gradio)
        for i, item in enumerate(outputs):
            try:
                if hasattr(item, "to_plotly_json"): # Es un gráfico de Plotly
                    # Gradio extrae la info del dict
                    d = item.to_plotly_json()
                    # Simular orjson/json de python
                    json.dumps(d, default=str)
                elif isinstance(item, (dict, list, str, int, float, bool, type(None))):
                    json.dumps(item, default=str)
                print(f"Item {i} serializable.")
            except Exception as e:
                print(f"❌ FALLO DE SERIALIZACIÓN EN EL ITEM {i}: {type(item)}")
                print(f"Error: {e}")
                sys.exit(1)
                
        print("\n✅ Todos los items son serializables. El problema no es de serialización JSON.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_serialization()
