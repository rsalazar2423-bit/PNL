"""
============================================================
Módulo: services/stages/base.py
============================================================
Clase base para el diseño modular de etapas del pipeline NLP.
"""

class PipelineStage:
    """
    Clase base abstracta que define el contrato de una etapa del pipeline.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, context: dict, progress=None) -> dict:
        """
        Ejecuta la lógica de la etapa sobre el contexto mutable del pipeline.
        """
        raise NotImplementedError
