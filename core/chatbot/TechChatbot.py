import os
import logging
from typing import List, Dict
from groq import Groq
from .EmbeddingManager import EmbeddingManager

logger = logging.getLogger(__name__)


class TechChatbot:
    """Chatbot especializado en buscar productos tecnológicos en descuento"""

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
                model="llama-3.3-70b-versatile",
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
        system_prompt = """Eres un asistente especializado en buscar productos tecnológicos en descuento 
        across múltiples tiendas. Eres un buscador inteligente, no el vendedor.

DIRECTRICES CRÍTICAS:
1. Eres un BUSCADOR que encuentra productos en diferentes tiendas, NO el vendedor
2. Siempre menciona la tienda de origen (ej: "Encontré en Alkosto...")
3. Para saludos simples: responde cordialmente sin mencionar productos
4. Solo muestra productos cuando la consulta es específica
5. Destaca los descuentos y precios rebajados
6. Sé preciso con especificaciones técnicas
7. Incluye enlaces y disponibilidad
8. Responde en español perfecto

FORMATO DE RESPUESTAS:
- Saludos: "¡Hola! 👋 Soy tu buscador de ofertas tech. ¿Qué producto necesitas?"
- Con productos: "📦 En Alkosto encontré [producto] por [precio] ([descuento])"
- Sin productos: "No encontré ofertas para '[consulta]'. ¿Podrías ser más específico?"

Ejemplos:
- "hola" → "¡Hola! 👋 ¿Buscas algún producto tecnológico en oferta?"
- "laptop i5" → "💻 En Alkosto encontré Lenovo IdeaPad 3 con i5 por $2,300,000 (15% off)"
- "no encuentro" → "¿Podrías decirme más características? 📏 ¿Qué RAM, almacenamiento o precio buscas?"
"""

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Agregar historial de conversación (últimos 3 mensajes)
        for msg in self.conversation_history[-6:]:  # 3 interacciones
            messages.append({
                "role": "user" if msg["type"] == "user" else "assistant",
                "content": msg["content"]
            })

        # Agregar contexto de productos si existe y es relevante
        if product_info and self._is_product_related_query(user_input):
            product_context = self._format_products_for_prompt(product_info)
            messages.append({
                "role": "system",
                "content": f"PRODUCTOS ENCONTRADOS EN TIENDAS:\n{product_context}\n\nMenciona siempre la tienda de origen y destaca los descuentos."
            })

        # Agregar el mensaje actual del usuario
        messages.append({"role": "user", "content": user_input})

        return messages

    def _format_products_for_prompt(self, products: List[Dict]) -> str:
        """Formatea productos para el prompt incluyendo la tienda de origen"""
        if not products:
            return "No hay productos disponibles para esta búsqueda."

        formatted_products = []
        for i, product in enumerate(products[:3]):  # Máximo 3 productos
            # Obtener la tienda (source) o usar "Alkosto" por defecto
            store = product.get('source', 'alkosto').upper()

            product_str = f"🏪 {store} - PRODUCTO {i + 1}:\n"
            product_str += f"   📦 Nombre: {product.get('name', 'Sin nombre')}\n"
            product_str += f"   🏷️ Marca: {product.get('brand', 'Sin marca')}\n"
            product_str += f"   💰 Precio: ${product.get('price', 0):,.0f}\n"

            # Destacar descuentos
            discount = product.get('discount_percent', '0%')
            if discount not in [None, '0%', '0']:
                product_str += f"   ⭐ Descuento: {discount} OFF\n"

            product_str += f"   📋 Categoría: {product.get('category', 'Sin categoría')}\n"

            # Agregar specs importantes
            specs = product.get('specifications', {})
            if specs:
                product_str += "   ⚙️ Especificaciones:\n"
                for key in ['RAM', 'Almacenamiento', 'Procesador', 'Pantalla', 'Memoria', 'Tarjeta gráfica']:
                    if key in specs:
                        product_str += f"     - {key}: {specs[key]}\n"

            product_str += f"   🌐 URL: {product.get('product_url', 'No disponible')}\n"
            product_str += f"   📸 Imagen: {product.get('image_url', 'No disponible')}\n"
            product_str += f"   📍 Disponibilidad: {product.get('availability', 'Disponible')}\n"

            formatted_products.append(product_str)

        return "\n" + "\n".join(formatted_products)

    def _is_product_related_query(self, user_input: str) -> bool:
        """Determina si la consulta está relacionada con productos"""
        general_phrases = [
            'hola', 'hello', 'hi', 'buenos días', 'buenas tardes', 'buenas noches',
            'qué tal', 'cómo estás', 'gracias', 'thanks', 'thank you', 'adiós',
            'chao', 'bye', 'saludos', 'help', 'ayuda', 'información'
        ]

        input_lower = user_input.lower().strip()

        # Si es solo un saludo o frase general, no buscar productos
        if any(phrase in input_lower for phrase in general_phrases):
            return False

        return True

    def _calculate_dynamic_threshold(self, user_input: str) -> float:
        """Calcula threshold dinámico basado en la consulta"""
        input_lower = user_input.lower()

        # Consultas generales/saludos - threshold alto
        general_words = ['hola', 'holi', 'hey', 'hi', 'hello', 'qué tal', 'cómo estás', 'gracias']
        if any(word in input_lower for word in general_words):
            return 0.8  # Muy alto para evitar resultados no relevantes

        # Consultas específicas - threshold medio
        specific_words = ['precio', 'cuesta', 'valor', 'costó', 'comprar', 'quiero', 'busco',
                          'necesito', 'recomienda', 'muestra', 'muéstrame', 'tienes', 'disponible']
        if any(word in input_lower for word in specific_words):
            return 0.45

        # Consultas técnicas - threshold bajo-medio
        tech_words = ['ram', 'procesador', 'almacenamiento', 'pantalla', 'memoria', 'gb', 'tb',
                      'intel', 'amd', 'ryzen', 'core', 'nvidia', 'graphics']
        if any(word in input_lower for word in tech_words):
            return 0.4

        return 0.5  # Default

    def _fallback_response(self, user_input: str, product_info: List[Dict] = None) -> str:
        """Respuesta de fallback si la API falla"""
        if product_info and self._is_product_related_query(user_input):
            product = product_info[0]
            store = product.get('source', 'alkosto').upper()
            return (
                f"En {store} encontré {product.get('name', 'un producto')} "
                f"de {product.get('brand', 'marca reconocida')} por ${product.get('price', 0):,.0f}. "
                f"¿Te interesa que busque más detalles?"
            )
        else:
            return "¡Hola! 👋 Soy tu buscador de ofertas tech. ¿Qué producto necesitas encontrar?"

    def chat(self, user_input: str) -> str:
        """Flujo completo de chat con embeddings + Groq"""
        try:
            logger.info(f"👤 Usuario: {user_input}")

            # 1. Determinar si buscar productos
            products = []
            if self._is_product_related_query(user_input):
                threshold = self._calculate_dynamic_threshold(user_input)
                products = self.embedding_manager.search_products(
                    user_input,
                    top_k=3,
                    threshold=threshold
                )
                logger.info(f"🔍 Encontrados {len(products)} productos con threshold {threshold}")

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
            self.conversation_history = self.conversation_history[-10:]

            logger.info(f"🤖 Asistente: {response}")
            return response

        except Exception as e:
            logger.error(f"❌ Error en chat: {e}")
            return "¡Disculpa! Estoy teniendo problemas técnicos. ¿Podrías intentarlo de nuevo?"

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