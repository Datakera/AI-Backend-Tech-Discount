import os
from huggingface_hub import login

# Dentro de ChatbotTrainer.__init__ o antes de cargar modelo
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token)
else:
    print("⚠️ No se encontró HUGGINGFACE_HUB_TOKEN en el entorno. Puede que falle la descarga del modelo.")
