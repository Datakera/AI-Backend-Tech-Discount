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
        """Genera respuesta usando Groq SDK con contexto de productos VALIDADOS"""
        try:
            if not self.client:
                return self._fallback_response(user_input, product_info)

            # ✅ VALIDACIÓN CRÍTICA: Si no hay productos relevantes, forzar respuesta de "no encontrado"
            if not self._has_relevant_products(user_input, product_info):
                return self._no_products_response(user_input)

            # Construir el mensaje con contexto
            messages = self._build_messages(user_input, product_info)

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3,  # ✅ Temperatura más baja para reducir invención
                max_tokens=800,  # ✅ Aumentado para que quepan más productos
                top_p=0.8
            )

            response = chat_completion.choices[0].message.content

            # ✅ VALIDACIÓN POST-RESPUESTA: Asegurar que solo menciona productos del contexto
            return self._validate_response(response, product_info)

        except Exception as e:
            logger.error(f"❌ Error con Groq API: {e}")
            return self._fallback_response(user_input, product_info)

    def _build_messages(self, user_input: str, product_info: List[Dict] = None) -> List[Dict]:
        """Construye los mensajes para la API de Groq"""

        available_stores = self._get_available_stores(product_info)
        stores_text = ", ".join(
            [store.capitalize() for store in available_stores]) if available_stores else "las tiendas disponibles"

        system_prompt = f"""Eres un asistente especializado en buscar productos tecnológicos en descuento. 
        Trabajas EXCLUSIVAMENTE con la información proporcionada en el contexto.

        REGLAS ABSOLUTAS (NO VIOLAR):
        1. SOLO menciona productos que estén en el contexto proporcionado
        2. SOLO menciona tiendas que estén en el contexto proporcionado  
        3. NUNCA inventes productos, precios, descuentos, especificaciones o tiendas
        4. Menciona los nombres de productos TAL CUAL aparecen en el contexto
        5. Incluye siempre la marca y modelo específico del producto
        6. Los enlaces y disponibilidad deben ser EXACTAMENTE los del contexto
        7. SIEMPRE ofrece ayuda adicional al final de tu respuesta
        8. Sé proactivo y amigable, como un buen asistente

        INFORMACIÓN DISPONIBLE ACTUALMENTE:
        - Tiendas: {stores_text}
        - Productos encontrados: {len(product_info) if product_info else 0}

        IMPORTANTE: 
        - Muestra los productos más relevantes (hasta 5) 
        - Al final de cada respuesta, ofrece tu ayuda para seguir buscando o para más detalles
        - Sé proactivo y ofrece ayuda para comparar, elegir o obtener más información

        EJEMPLO CORRECTO:
        "💻 En Alkosto encontré 'Computador Portátil Gamer HP Victus 15.6 Pulgadas Fb2024la AMD Ryzen 5' por $5,399,000. 
        ¿Te gustaría que te ayude a comparar modelos o necesitas más información sobre este?"

        EJEMPLO INCORRECTO:
        "💻 En Alkosto encontré 'Computador Portátil Gamer HP Victus 15.6 Pulgadas Fb2024la AMD Ryzen 5' por $5,399,000."
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
                "content": f"PRODUCTOS ENCONTRADOS EN TIENDAS:\n{product_context}\n\n"
                           f"IMPORTANTE: Menciona siempre la tienda de origen, destaca los descuentos, "
                           f"y al final ofrece ayuda para elegir o comparar productos."
            })

        # Agregar el mensaje actual del usuario
        messages.append({"role": "user", "content": user_input})

        return messages

    def _format_products_for_prompt(self, products: List[Dict]) -> str:
        """Formatea productos para el prompt de manera más concisa"""
        if not products:
            return "No hay productos disponibles para esta búsqueda."

        formatted_products = []
        for i, product in enumerate(products[:5]):  # Hasta 5 productos
            store = product.get('source', 'alkosto').upper()

            # Formato más conciso
            product_str = f"{store} - {product.get('name', 'Sin nombre')}"
            product_str += f" | ${product.get('price', 0):,.0f}"

            discount = product.get('discount_percent', '0%')
            if discount not in [None, '0%', '0']:
                product_str += f" ({discount} OFF)"

            formatted_products.append(product_str)

        return "PRODUCTOS DISPONIBLES:\n" + "\n".join(formatted_products)

    def _get_available_stores(self, product_info: List[Dict] = None) -> List[str]:
        """Obtiene las tiendas disponibles DINÁMICAMENTE de la base de datos"""
        if not product_info:
            return []

        # Extraer tiendas únicas de los productos encontrados
        stores = set()
        for product in product_info:
            store = product.get('source', '').lower()
            if store:
                stores.add(store)

        return list(stores)

    def _get_available_stores_info(self) -> str:
        """Obtiene información de las tiendas disponibles en la base de datos"""
        try:
            # Intentar obtener productos del índice
            if hasattr(self.embedding_manager, 'get_all_products_from_index'):
                all_products = self.embedding_manager.get_all_products_from_index()
            else:
                # Fallback: usar los productos del historial de búsqueda reciente
                all_products = []
                for msg in self.conversation_history:
                    if msg.get("products_found", 0) > 0:
                        # Aquí necesitarías una manera de obtener los productos reales
                        pass
                return "Actualmente trabajo con varias tiendas de tecnología. ¿Te interesa buscar algún producto específico?"

            if not all_products:
                return "Actualmente no tengo información de productos en mi base de datos."

            # Extraer tiendas únicas
            stores = set()
            for product in all_products:
                store = product.get('source', '').strip().lower()
                if store:
                    stores.add(store)

            if not stores:
                return "Tengo productos de varias tiendas tecnológicas. ¿Qué producto estás buscando?"

            # Formatear respuesta
            store_list = [store.capitalize() for store in stores]
            store_list.sort()

            if len(store_list) == 1:
                return f"Actualmente tengo productos de {store_list[0]} en mi base de datos. ¿Te interesa buscar algo específico?"
            else:
                stores_text = ", ".join(store_list[:-1]) + f" y {store_list[-1]}"
                return f"Actualmente tengo productos de {stores_text} en mi base de datos. ¿Qué producto te interesa?"

        except Exception as e:
            logger.error(f"Error obteniendo tiendas disponibles: {e}")
            return "Tengo acceso a varias tiendas de tecnología. ¿En qué producto te puedo ayudar a buscar?"

    def _is_store_related_query(self, user_input: str) -> bool:
        """Determina si la consulta es sobre tiendas disponibles"""
        input_lower = user_input.lower().strip()

        store_keywords = [
            'tiendas', 'tienda', 'store', 'stores', 'dónde', 'donde',
            'qué tiendas', 'que tiendas', 'de qué tiendas', 'de que tiendas',
            'en qué tiendas', 'en que tiendas', 'qué almacenes', 'que almacenes',
            'dónde buscar', 'donde buscar', 'qué empresas', 'que empresas',
            'qué marcas', 'que marcas', 'qué sitios', 'que sitios', 'dónde comprar', 'donde comprar'
        ]

        # Consultas específicas sobre tiendas
        specific_store_queries = [
            'de que tiendas', 'que tiendas', 'qué tiendas', 'tiendas tienes',
            'tiendas tiene', 'tiendas hay', 'tiendas disponibles', 'tiendas trabajas',
            'tiendas manejas', 'en qué almacenes', 'qué empresas'
        ]

        # Si contiene palabras clave específicas de tiendas
        if any(keyword in input_lower for keyword in specific_store_queries):
            return True

        # Si es una consulta muy general que podría ser sobre tiendas
        general_store_indicators = ['tienda', 'store', 'almacén', 'empresa']
        if (any(word in input_lower for word in general_store_indicators) and
                len(input_lower.split()) <= 4):  # Consultas cortas
            return True

        return False

    def _validate_response(self, response: str, product_info: List[Dict] = None) -> str:
        """Valida que la respuesta solo mencione productos de la base de datos ACTUAL"""
        if not product_info:
            return self._no_products_response("")

        response_lower = response.lower()

        # Detectar si menciona palabras clave de productos
        is_talking_about_products = any(word in response_lower for word in [
            'encontré', 'encontre', 'producto', 'tenemos', 'ofertas', 'disponible', 'precio', 'victus', 'hp',
            'computador'
        ])

        # ✅ Validación MÁS INTELIGENTE: Buscar coincidencias parciales
        is_mentioning_real_products = False
        mentioned_products_count = 0

        for product in product_info:
            product_name = product.get('name', '').lower()
            brand = product.get('brand', '').lower()

            # ✅ Coincidencia parcial: Si el nombre del producto está contenido en la respuesta
            # o si la respuesta contiene palabras clave del producto
            name_match = any(keyword in response_lower for keyword in product_name.split()[:5])  # Primeras 5 palabras
            brand_match = brand and brand in response_lower

            if name_match or brand_match:
                is_mentioning_real_products = True
                mentioned_products_count += 1
                logger.info(f"   ✅ Coincidencia: {product_name[:50]}...")

        # ✅ Validación de tiendas
        is_mentioning_real_stores = False
        for product in product_info:
            store = product.get('source', '').lower()
            if store and store in response_lower:
                is_mentioning_real_stores = True
                break

        logger.info(f"   📊 Productos mencionados: {mentioned_products_count}/{len(product_info)}")
        logger.info(f"   📊 Habla de productos: {is_talking_about_products}")
        logger.info(f"   📊 Menciona productos reales: {is_mentioning_real_products}")
        logger.info(f"   📊 Menciona tiendas reales: {is_mentioning_real_stores}")

        # ✅ Lógica de validación mejorada
        if is_talking_about_products:
            if not is_mentioning_real_products:
                logger.warning("⚠️ Chatbot habla de productos pero no menciona los reales")
                return self._no_products_response("")

            if not is_mentioning_real_stores and any(word in response_lower for word in ['en ', 'de ', 'tienda']):
                logger.warning("⚠️ Chatbot habla de tiendas pero no menciona las reales")
                return self._no_products_response("")

        # ✅ Si menciona al menos 1 producto real, la respuesta es válida
        if mentioned_products_count > 0:
            logger.info("   ✅ Respuesta validada correctamente")
            return response
        else:
            logger.warning("   ⚠️ No se detectaron productos reales en la respuesta")
            return self._no_products_response("")

    def _has_relevant_products(self, user_input: str, product_info: List[Dict] = None) -> bool:
        """Determina si los productos encontrados son realmente relevantes para la consulta"""
        if not product_info:
            return False

        # Umbral de similitud ajustado
        min_similarity = 0.45

        # Verificar relevancia semántica y contextual
        relevant_count = 0
        for product in product_info:
            similarity = product.get('similarity_score', 0)

            # Producto es relevante si tiene buena similitud
            if similarity >= min_similarity:
                relevant_count += 1

        logger.info(f"📊 Productos con similitud >= {min_similarity}: {relevant_count}/{len(product_info)}")

        # También considerar si la consulta es muy general vs productos encontrados
        if relevant_count == 0 and len(product_info) > 0:
            # Consulta muy general pero tenemos productos
            if len(user_input.split()) <= 2:  # Consultas cortas como "laptop", "celular"
                return True  # Mostrar resultados aunque similitud sea baja

        return relevant_count > 0

    def _no_products_response(self, user_input: str) -> str:
        """Respuesta cuando no hay productos relevantes en la base de datos"""

        # Respuesta especial para saludos
        if user_input.lower() in ['hola', 'hello', 'hi', 'buenos días', 'buenas tardes', 'buenas noches']:
            return "¡Hola! 👋 Soy tu asistente especializado en buscar productos tecnológicos en oferta. ¿En qué puedo ayudarte hoy? ¿Buscas algún producto específico?"

        suggestions = [
            "Intenta ser más específico con el modelo o características",
            "Prueba con otras palabras clave o marcas",
            "Revisa si hay errores de escritura en tu búsqueda",
            "¿Podrías darme más detalles sobre lo que necesitas?"
        ]

        import random
        suggestion = random.choice(suggestions)

        if user_input:
            return f"🔍 No encontré resultados para '{user_input}' en mi base de datos actual. {suggestion}"
        else:
            return f"🔍 No encontré productos que coincidan con tu búsqueda. {suggestion}"

    def _is_product_related_query(self, user_input: str) -> bool:
        """Determina si la consulta está relacionada con productos de manera inteligente"""
        input_lower = user_input.lower().strip()

        # Palabras que indican claramente NO es búsqueda de productos (conversación normal)
        conversation_phrases = [
            'hola', 'hello', 'hi', 'buenos días', 'buenas tardes', 'buenas noches',
            'qué tal', 'cómo estás', 'cómo te va', 'qué hay', 'qué onda',
            'gracias', 'thanks', 'thank you', 'adiós', 'chao', 'bye',
            'saludos', 'ok', 'vale', 'entendido', 'de nada', 'perdón', 'disculpa',
            'cómo estás hoy', 'qué cuentas', 'cómo ha estado', 'qué me cuentas'
        ]

        # 1. Si es EXACTAMENTE una frase de conversación → NO buscar productos
        if any(input_lower == phrase for phrase in conversation_phrases):
            return False

        # 2. Si contiene palabras de conversación general (aunque tenga otras palabras)
        general_conversation_words = [
            'cómo estás', 'qué tal', 'cómo te va', 'gracias', 'hola', 'buenos días',
            'buenas tardes', 'buenas noches', 'adiós', 'chao', 'bye'
        ]

        # Si contiene palabras de conversación y es una frase corta
        if (any(phrase in input_lower for phrase in general_conversation_words) and
                len(input_lower.split()) <= 4):
            return False

        # 3. Si contiene palabras de intención de búsqueda → SÍ buscar
        search_intent_words = [
            'buscar', 'busco', 'encontrar', 'encuentra', 'quiero', 'necesito',
            'recomienda', 'muestra', 'muéstrame', 'dime', 'ayuda', 'ayúdame',
            'producto', 'productos', 'oferta', 'ofertas', 'descuento', 'comprar',
            'laptop', 'celular', 'tablet', 'televisor', 'monitor', 'audífonos',
            'precio', 'cuesta', 'valor', 'costó', 'disponible', 'tienes'
        ]

        if any(word in input_lower for word in search_intent_words):
            return True

        # 4. Si contiene nombres de categorías/marcas comunes → SÍ buscar
        tech_keywords = [
            'samsung', 'apple', 'iphone', 'lenovo', 'hp', 'dell', 'asus', 'acer',
            'portátil', 'portatil', 'laptop', 'notebook', 'smartphone', 'celular',
            'tablet', 'ipad', 'tv', 'televisor', 'monitor', 'proyector', 'consola',
            'playstation', 'xbox', 'nintendo', 'audífonos', 'headphones', 'impresora'
        ]

        if any(keyword in input_lower for keyword in tech_keywords):
            return True

        # 5. Consultas muy cortas sin contexto → NO buscar
        if len(input_lower.split()) <= 2 and not any(word in input_lower for word in tech_keywords):
            return False

        # 6. Default: buscar por si acaso
        return True

    def _calculate_dynamic_threshold(self, user_input: str) -> float:
        """Calcula threshold dinámico basado en la consulta"""
        input_lower = user_input.lower()

        # Consultas que mezclan saludo + búsqueda → threshold medio
        mixed_phrases = [
            'hola me podrías ayudar', 'buenos días quiero', 'hola busco',
            'hola necesito', 'hola quiero', 'buenas tardes me recomiendas'
        ]

        if any(phrase in input_lower for phrase in mixed_phrases):
            return 0.4  # Threshold medio para consultas mixtas

        # Consultas técnicas específicas → threshold bajo
        tech_specific = ['ram', 'procesador', 'almacenamiento', 'pantalla',
                         'gb', 'tb', 'intel', 'amd', 'ryzen', 'core', 'nvidia']
        if any(word in input_lower for word in tech_specific):
            return 0.35

        # Consultas generales de producto → threshold medio
        return 0.45

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

            # 1. ✅ PRIMERO: Verificar si es consulta sobre tiendas
            if self._is_store_related_query(user_input):
                store_info = self._get_available_stores_info()
                response = f"🏪 {store_info} ¿Te interesa buscar algún producto en particular?"

                # Guardar en historial
                self.conversation_history.append({
                    "type": "user",
                    "content": user_input,
                    "products_found": 0
                })
                self.conversation_history.append({
                    "type": "assistant",
                    "content": response
                })

                logger.info(f"🤖 Asistente: {response}")
                return response

            # 2. Determinar si buscar productos
            products = []
            should_search = self._is_product_related_query(user_input)

            if should_search:
                threshold = self._calculate_dynamic_threshold(user_input)
                products = self.embedding_manager.search_products(
                    user_input,
                    top_k=5,
                    threshold=0.3
                )

                # Filtrar por relevancia
                products = [p for p in products if p.get('similarity_score', 0) >= 0.4]
                logger.info(f"🔍 Productos después de filtrado: {len(products)}")

                for i, product in enumerate(products):
                    logger.info(
                        f"   {i + 1}. {product.get('name')} - Score: {product.get('similarity_score', 0):.3f} - Tienda: {product.get('source')}")

            # 3. Generar respuesta apropiada
            if not should_search:
                # Es conversación normal, usar Groq para respuesta natural
                response = self._generate_conversational_response(user_input)
            elif not self._has_relevant_products(user_input, products):
                # Búsqueda sin resultados relevantes
                response = self._no_products_response(user_input)
            else:
                # Búsqueda con resultados, generar respuesta con productos
                response = self.generate_response(user_input, products)

            # 4. Guardar en historial
            self.conversation_history.append({
                "type": "user",
                "content": user_input,
                "products_found": len(products) if should_search else 0
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

    def _generate_conversational_response(self, user_input: str) -> str:
        """Genera respuestas para conversación normal (no búsqueda de productos)"""
        try:
            if not self.client:
                return "¡Hola! 👋 ¿En qué puedo ayudarte hoy?"

            messages = [
                {"role": "system", "content": """Eres un asistente amigable y conversacional especializado en productos tecnológicos. 
                Responde de manera natural y cordial a saludos y conversación general.
                Mantén tus respuestas breves y amigables.
                Si es apropiado, pregunta si la persona necesita ayuda con productos tecnológicos."""},
                {"role": "user", "content": user_input}
            ]

            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=150,
                top_p=0.9
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Error en respuesta conversacional: {e}")
            return "¡Hola! 👋 ¿En qué puedo ayudarte hoy?"

    def quick_test(self, test_query: str = "hola") -> str:
        """Prueba rápida del chatbot"""
        try:
            return self.chat(test_query)
        except Exception as e:
            return f"Error en prueba: {e}"