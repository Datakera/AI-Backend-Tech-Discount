import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from typing import List, Dict, Optional
from .EmbeddingManager import EmbeddingManager
import re

logger = logging.getLogger(__name__)


class TechChatbot:
    """Chatbot principal que combina búsqueda semántica con respuestas generativas"""

    def __init__(self,
                 base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 lora_path: str = "models/chatbot_lora"):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Componentes
        self.tokenizer = None
        self.model = None
        self.embedding_manager = EmbeddingManager()

        # Estado de conversación
        self.conversation_history = []
        self.max_history = 5

        logger.info(f"🤖 Inicializando TechChatbot en {self.device}")

    def load_model(self, load_base_only: bool = False):
        """Carga el modelo base y LoRA adapter"""
        try:
            logger.info("🔄 Cargando modelo...")

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if not load_base_only and os.path.exists(self.lora_path):
                # Cargar modelo base + LoRA
                logger.info("📦 Cargando modelo con LoRA fine-tuning...")

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )

                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_path,
                    torch_dtype=torch.float16
                )

                logger.info("✅ Modelo LoRA cargado correctamente")
            else:
                # Cargar solo modelo base
                logger.info("📦 Cargando modelo base (sin fine-tuning)...")

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )

                logger.info("✅ Modelo base cargado correctamente")

            self.model.eval()

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def _extract_search_intent(self, user_input: str) -> Dict:
        """Extrae la intención de búsqueda del input del usuario"""
        intent = {
            'query': user_input.lower(),
            'product_type': None,
            'brand': None,
            'price_range': {'min': None, 'max': None},
            'looking_for_deals': False,
            'category': None
        }

        # Detectar búsqueda de ofertas
        deal_keywords = ['oferta', 'descuento', 'barato', 'precio', 'promocion', 'rebaja']
        if any(keyword in intent['query'] for keyword in deal_keywords):
            intent['looking_for_deals'] = True

        # Detectar categorías comunes
        categories = {
            'celulares': ['celular', 'telefono', 'smartphone', 'movil'],
            'computadores': ['computador', 'pc', 'laptop', 'portatil', 'notebook'],
            'televisores': ['tv', 'television', 'televisor', 'pantalla'],
            'audífonos': ['audifonos', 'auriculares', 'headphones'],
            'tablets': ['tablet', 'ipad'],
            'electrodomésticos': ['nevera', 'lavadora', 'microondas', 'licuadora']
        }

        for category, keywords in categories.items():
            if any(keyword in intent['query'] for keyword in keywords):
                intent['category'] = category
                intent['product_type'] = category
                break

        # Detectar marcas comunes
        brands = ['samsung', 'apple', 'huawei', 'xiaomi', 'lg', 'sony',
                  'lenovo', 'hp', 'dell', 'asus', 'acer', 'nokia']

        for brand in brands:
            if brand in intent['query']:
                intent['brand'] = brand
                break

        # Detectar rangos de precio
        price_patterns = [
            r'menos de (\d+)',
            r'bajo (\d+)',
            r'entre (\d+) y (\d+)',
            r'(\d+) a (\d+)',
            r'máximo (\d+)',
            r'hasta (\d+)'
        ]

        for pattern in price_patterns:
            match = re.search(pattern, intent['query'])
            if match:
                if len(match.groups()) == 1:
                    intent['price_range']['max'] = int(match.group(1)) * 1000
                elif len(match.groups()) == 2:
                    intent['price_range']['min'] = int(match.group(1)) * 1000
                    intent['price_range']['max'] = int(match.group(2)) * 1000

        return intent

    def _search_products(self, intent: Dict, top_k: int = 5) -> List[Dict]:
        """Busca productos basado en la intención extraída"""
        try:
            # Si hay una consulta específica, usar búsqueda semántica
            if intent['product_type'] or intent['category']:
                query = f"{intent['category'] or intent['product_type']}"
                if intent['brand']:
                    query += f" {intent['brand']}"
                if intent['looking_for_deals']:
                    query += " oferta descuento"

                results = self.embedding_manager.search_products(query, top_k=top_k)
            else:
                # Búsqueda por filtros
                results = self.embedding_manager.search_by_category_and_price(
                    category=intent['category'],
                    min_price=intent['price_range']['min'],
                    max_price=intent['price_range']['max'],
                    with_discount=intent['looking_for_deals'],
                    top_k=top_k
                )

            # Filtrar por marca si se especifica
            if intent['brand'] and results:
                results = [r for r in results if intent['brand'].lower() in r['name'].lower()
                           or intent['brand'].lower() in r['brand'].lower()]

            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda de productos: {e}")
            return []

    def _format_products_for_context(self, products: List[Dict]) -> str:
        """Formatea productos para incluir en el contexto del modelo"""
        if not products:
            return "No se encontraron productos que coincidan con tu búsqueda."

        context = "Productos encontrados:\n"
        for i, product in enumerate(products[:3], 1):  # Máximo 3 productos
            price_info = f"${product['price']:,.0f}"
            discount_info = ""

            if product.get('discount_percent', '0%') != '0%':
                discount_info = f" (¡{product['discount_percent']} de descuento!)"

            context += f"{i}. {product['name']} - {product['brand']} - {price_info}{discount_info}\n"

            # Agregar especificaciones relevantes
            specs = product.get('specifications', {})
            if specs:
                important_specs = []
                for key, value in list(specs.items())[:2]:
                    important_specs.append(f"{key}: {value}")
                if important_specs:
                    context += f"   Características: {', '.join(important_specs)}\n"

        return context

    def _generate_response(self, user_input: str, products_context: str) -> str:
        """Genera respuesta usando el modelo de lenguaje"""
        try:
            # Crear prompt con contexto
            system_prompt = """Eres un asistente de ventas especializado en productos tecnológicos de Colombia. 
Eres amigable, conocedor y ayudas a los usuarios a encontrar los mejores productos.
Responde de manera conversacional y natural, mencionando precios, descuentos y características relevantes.
Si hay productos disponibles, recomiéndalos. Si no hay productos, sugiere alternativas o pide más detalles."""

            conversation_context = ""
            if self.conversation_history:
                conversation_context = "\nConversación previa:\n"
                for exchange in self.conversation_history[-2:]:  # Últimas 2 interacciones
                    conversation_context += f"Usuario: {exchange['user']}\nAsistente: {exchange['assistant']}\n"

            prompt = f"""<s>[INST] {system_prompt}

{products_context}

{conversation_context}

Usuario: {user_input}

Responde de manera amigable y útil, mencionando los productos específicos cuando sea relevante. [/INST]"""

            # Tokenizar
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decodificar respuesta
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extraer solo la respuesta del asistente
            response_start = response.find("[/INST]") + 7
            if response_start > 6:
                response = response[response_start:].strip()

            return response

        except Exception as e:
            logger.error(f"❌ Error generando respuesta: {e}")
            return "Lo siento, tuve un problema generando la respuesta. ¿Podrías intentar de nuevo?"

    def chat(self, user_input: str) -> str:
        """Función principal de chat"""
        try:
            # 1. Extraer intención de búsqueda
            intent = self._extract_search_intent(user_input)

            # 2. Buscar productos relevantes
            products = self._search_products(intent, top_k=5)

            # 3. Formatear contexto de productos
            products_context = self._format_products_for_context(products)

            # 4. Generar respuesta
            response = self._generate_response(user_input, products_context)

            # 5. Actualizar historial de conversación
            self.conversation_history.append({
                'user': user_input,
                'assistant': response,
                'products_found': len(products)
            })

            # Mantener solo las últimas conversaciones
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return response

        except Exception as e:
            logger.error(f"❌ Error en chat: {e}")
            return "Lo siento, tuve un problema procesando tu consulta. ¿Podrías intentar de nuevo?"

    def get_conversation_stats(self) -> Dict:
        """Obtiene estadísticas de la conversación actual"""
        if not self.conversation_history:
            return {"total_interactions": 0}

        total_interactions = len(self.conversation_history)
        successful_searches = sum(1 for conv in self.conversation_history if conv.get('products_found', 0) > 0)

        return {
            "total_interactions": total_interactions,
            "successful_searches": successful_searches,
            "success_rate": f"{(successful_searches / total_interactions * 100):.1f}%" if total_interactions > 0 else "0%"
        }

    def clear_conversation(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
        logger.info("🧹 Historial de conversación limpiado")

    def save_conversation(self, filename: str = None):
        """Guarda la conversación actual"""
        if not filename:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        filepath = os.path.join("data/conversations", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'conversation_history': self.conversation_history,
                'stats': self.get_conversation_stats()
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Conversación guardada en: {filepath}")
        return filepath