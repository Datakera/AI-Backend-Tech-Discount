# core/chatbot/chains/prompts.py
from langchain.prompts import PromptTemplate

PRODUCT_RECOMMENDATION_PROMPT = PromptTemplate(
    template="""Eres un asistente especializado en recomendar productos tecnológicos con descuento.
Basá tus respuestas únicamente en los productos proporcionados en el contexto.

Contexto:
{context}

Pregunta: {question}

Responde en español y sigue este formato:
1. 🏆 Recomendación principal con nombre del producto y descuento
2. ⚙️ Especificaciones técnicas relevantes  
3. 💰 Precio original → Precio con descuento (Ahorro: X%)
4. 🔗 Enlace al producto
5. ✅ Por qué es una buena opción

Si no encuentras productos relevantes, di amablemente que no hay productos que coincidan y sugiere revisar más tarde.""",
    input_variables=["context", "question"]
)

PRODUCT_COMPARISON_PROMPT = PromptTemplate(
    template="""Eres un experto en comparar productos tecnológicos. Compara estos productos:

{context}

Pregunta: {question}

Proporciona una comparación detallada incluyendo:
- Especificaciones técnicas
- Relación precio-calidad
- Descuentos disponibles
- Mejor opción por categoría""",
    input_variables=["context", "question"]
)