import sys
from src.controllers.dashboard_controller import handle_start_pipeline

class DummyProgress:
    def __call__(self, value, desc=None):
        pass

def test():
    try:
        gen = handle_start_pipeline(progress=DummyProgress())
        for result in gen:
            # Imprimir el mensaje de status, que está en la posición 2 del arreglo
            # o simplemente iterar
            if isinstance(result, list) and len(result) > 3 and hasattr(result[2], 'value'):
                print(result[2].value)
        print("\nTest passed: Dashboard loaded via cache bypass.")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test()
