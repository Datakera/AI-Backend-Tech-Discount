import os
import logging
from typing import List, Dict
from groq import Groq
from .EmbeddingManager import EmbeddingManager

logger = logging.getLogger(__name__)


class TechChatbot:
    """Chatbot usando Groq SDK oficial con embeddings locales"""

    def __init__(self, groq_api_key: str = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.embedding_manager = EmbeddingManager()
        self.conversation_history = []

        if not self.groq_api_key:
            logger.warning("⚠️ GROQ_API_KEY no encontrada. Usa environment variable o pásala al constructor.")
            self.client = None
        else:
            self.client = Groq(api_key=self.groq_api_key)

    def generate_response(self, user_input: str, product_info: List[Dict] = None) -> str:
        """Genera respuesta usando Groq SDK con contexto de productos"""
        try:
            if not self.client:
                return self._fallback_response(user_input, product_info)

            # Construir el mensaje con contexto
            messages = self._build_messages(user_input, product_info)

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",  # ✅ Modelo mejorado
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            logger.error(f"❌ Error con Groq API: {e}")
            return self._fallback_response(user_input, product_info)

    def _build_messages(self, user_input: str, product_info: List[Dict] = None) -> List[Dict]:
        """Construye los mensajes para la API de Groq"""
        system_prompt = """Eres un asistente virtual especializado en productos tecnológicos de una tienda online.
Eres amable, profesional y servicial. Usa emojis apropiados y mantén un tono conversacional.

DIRECTRICES IMPORTANTES:
1. Responde en español perfecto
2. Sé conciso pero informativo (máximo 2-3 párrafos)
3. Si hay productos relevantes, menciónalos naturalmente con sus características
4. Incluye URLs e imágenes cuando sea relevante
5. Si no hay productos exactos, sugiere alternativas similares
6. Mantén un tono entusiasta pero profesional
7. Usa formato de texto amigable (no markdown)

Ejemplo de respuestas buenas:
- "¡Perfecto! Tengo este Samsung Galaxy S23 por $2,500,000 con 256GB 💫"
- "No encontré exactamente lo que buscas, pero te recomiendo estas alternativas similares..."
- "¡Hola! 👋 ¿Buscas algún producto tecnológico en especial hoy?"
"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Agregar historial de conversación (últimos 4 mensajes)
        for msg in self.conversation_history[-8:]:  # 4 interacciones
            messages.append({
                "role": "user" if msg["type"] == "user" else "assistant",
                "content": msg["content"]
            })

        # Agregar contexto de productos si existe
        if product_info:
            product_context = self._format_products_for_prompt(product_info)
            messages.append({
                "role": "system",
                "content": f"CONTEXTO DE PRODUCTOS DISPONIBLES:\n{product_context}\n\nResponde mencionando los productos más relevantes de forma natural."
            })

        # Agregar el mensaje actual del usuario
        messages.append({"role": "user", "content": user_input})

        return messages

    def _format_products_for_prompt(self, products: List[Dict]) -> str:
        """Formatea productos para el prompt de manera eficiente"""
        if not products:
            return "No hay productos disponibles que coincidan con la búsqueda."

        formatted_products = []
        for i, product in enumerate(products[:4]):  # Máximo 4 productos
            product_str = f"🎯 PRODUCTO {i + 1}:\n"
            product_str += f"   Nombre: {product.get('name', 'Sin nombre')}\n"
            product_str += f"   Marca: {product.get('brand', 'Marca no especificada')}\n"
            product_str += f"   Precio: ${product.get('price', 0):,.0f}\n"

            if product.get('discount_percent') not in [None, '0%', '0']:
                product_str += f"   Descuento: {product.get('discount_percent')}\n"

            product_str += f"   Categoría: {product.get('category', 'Sin categoría')}\n"

            # Agregar specs importantes
            specs = product.get('specifications', {})
            if specs:
                product_str += "   Especificaciones:\n"
                for key in ['RAM', 'Almacenamiento', 'Procesador', 'Pantalla', 'Memoria']:
                    if key in specs:
                        product_str += f"     - {key}: {specs[key]}\n"

            product_str += f"   URL: {product.get('product_url', 'No disponible')}\n"
            product_str += f"   Imagen: {product.get('image_url', 'No disponible')}\n"

            formatted_products.append(product_str)

        return "\n" + "\n".join(formatted_products)

    def _fallback_response(self, user_input: str, product_info: List[Dict] = None) -> str:
        """Respuesta de fallback si la API falla"""
        if product_info:
            product = product_info[0]
            return (
                f"¡Hola! Encontré {product.get('name', 'un producto')} de {product.get('brand', 'marca reconocida')} "
                f"por ${product.get('price', 0):,.0f}. ¿Te interesa que te dé más detalles o busco otras opciones?"
            )
        else:
            return "¡Hola! 👋 Soy tu asistente de tecnología. ¿En qué puedo ayudarte hoy? Puedo buscarte productos tecnológicos, comparar precios o mostrarte ofertas."

    def chat(self, user_input: str) -> str:
        """Flujo completo de chat con embeddings + Groq"""
        try:
            logger.info(f"👤 Usuario: {user_input}")

            # 1. Buscar productos relevantes
            products = self.embedding_manager.search_products(user_input, top_k=4, threshold=0.25)

            # 2. Generar respuesta con Groq
            response = self.generate_response(user_input, products)

            # 3. Guardar en historial
            self.conversation_history.append({
                "type": "user",
                "content": user_input,
                "products_found": len(products)
            })
            self.conversation_history.append({
                "type": "assistant",
                "content": response
            })

            # Limitar historial para no exceder contexto
            self.conversation_history = self.conversation_history[-12:]

            logger.info(f"🤖 Asistente: {response}")
            return response

        except Exception as e:
            logger.error(f"❌ Error en chat: {e}")
            return "¡Disculpa! Estoy teniendo problemas técnicos momentáneos. ¿Podrías intentarlo de nuevo en un momento?"

    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []

    def get_chat_stats(self) -> Dict:
        """Estadísticas de la conversación"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for msg in self.conversation_history if msg["type"] == "user"),
            "assistant_messages": sum(1 for msg in self.conversation_history if msg["type"] == "assistant"),
            "last_products_found": self.conversation_history[-2]["products_found"] if len(
                self.conversation_history) >= 2 else 0
        }

    def quick_test(self, test_query: str = "hola") -> str:
        """Prueba rápida del chatbot"""
        try:
            return self.chat(test_query)
        except Exception as e:
            return f"Error en prueba: {e}"