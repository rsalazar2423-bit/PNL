import sys
from src.controllers.dashboard_controller import load_dashboard_ui
import gradio as gr
from gradio.utils import json_dumps

def test_gradio_serialization():
    try:
        outputs = load_dashboard_ui()
        print(f"✅ Dashboard cargado. {len(outputs)} elementos.")
        
        # Test serialization using Gradio's internal json_dumps
        for i, item in enumerate(outputs):
            try:
                # gradio json_dumps handles dicts, gr.update(), objects, etc.
                if hasattr(item, "to_plotly_json"):
                    item = item.to_plotly_json()
                elif hasattr(item, "dict"):
                    item = item.dict()
                json_dumps(item)
                print(f"Item {i} serializable via Gradio.")
            except Exception as e:
                print(f"❌ FALLO DE SERIALIZACIÓN EN EL ITEM {i}: {type(item)}")
                print(f"Error: {e}")
                sys.exit(1)
                
        print("\n✅ Todos los items son serializables por Gradio json_dumps.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_gradio_serialization()
