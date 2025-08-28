import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from typing import List, Dict
from .EmbeddingManager import EmbeddingManager
import re

logger = logging.getLogger(__name__)


class TechChatbot:
    """Chatbot para productos tecnológicos con búsqueda semántica"""

    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium",
                 lora_path: str = "models/tech_chatbot"):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.embedding_manager = EmbeddingManager()

        self.conversation_history = []
        self.max_history = 5

        logger.info(f"Inicializando chatbot en {self.device}")

    def load_model(self, load_base_only: bool = False):
        """Carga el modelo"""
        try:
            logger.info("Cargando modelo...")

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            if not load_base_only and os.path.exists(self.lora_path):
                # Cargar modelo base + LoRA
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )

                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.lora_path,
                    torch_dtype=torch.float32
                )
                logger.info("✅ Modelo con fine-tuning cargado")
            else:
                # Solo modelo base
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logger.info("✅ Modelo base cargado")

            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            self.model.eval()

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def _extract_search_intent(self, user_input: str) -> Dict:
        """Extrae intención de búsqueda"""
        intent = {
            'query': user_input.lower(),
            'category': None,
            'brand': None,
            'looking_for_deals': False
        }

        # Detectar categorías
        categories = {
            'celulares': ['celular', 'teléfono', 'smartphone', 'móvil'],
            'computadores': ['computador', 'pc', 'laptop', 'portátil', 'notebook'],
            'televisores': ['tv', 'televisor', 'televisión', 'pantalla'],
            'tablets': ['tablet', 'ipad'],
            'audífonos': ['audífonos', 'auriculares', 'headphones']
        }

        for category, keywords in categories.items():
            if any(keyword in intent['query'] for keyword in keywords):
                intent['category'] = category
                break

        # Detectar marcas
        brands = ['samsung', 'apple', 'xiaomi', 'huawei', 'lg', 'sony', 'lenovo', 'hp', 'dell']
        for brand in brands:
            if brand in intent['query']:
                intent['brand'] = brand
                break

        # Detectar ofertas
        deal_keywords = ['oferta', 'descuento', 'barato', 'promoción', 'rebaja']
        if any(keyword in intent['query'] for keyword in deal_keywords):
            intent['looking_for_deals'] = True

        return intent

    def _search_products(self, intent: Dict, top_k: int = 3) -> List[Dict]:
        """Busca productos basado en la intención"""
        try:
            if intent['category']:
                query = intent['category']
                if intent['brand']:
                    query += f" {intent['brand']}"
                if intent['looking_for_deals']:
                    query += " oferta descuento"

                return self.embedding_manager.search_products(query, top_k=top_k)
            else:
                return self.embedding_manager.search_by_category_and_price(
                    category=intent['category'],
                    with_discount=intent['looking_for_deals'],
                    top_k=top_k
                )
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []

    def _format_products_context(self, products: List[Dict]) -> str:
        """Formatea productos para el contexto"""
        if not products:
            return "No encontré productos específicos, pero puedo ayudarte con información general."

        context = "Productos relevantes:\n"
        for i, product in enumerate(products[:3], 1):
            price = product.get('price', 0)
            discount = product.get('discount_percent', '0%')

            context += f"{i}. {product['name']} - {product['brand']} - ${price:,.0f}"
            if discount != '0%':
                context += f" ({discount} descuento)"
            context += "\n"

        return context

    def _generate_response(self, user_input: str, products_context: str) -> str:
        """Genera respuesta usando el modelo"""
        try:
            # Crear prompt
            prompt = self._create_prompt(user_input, products_context)

            # Tokenizar
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)

            # Generar
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decodificar y limpiar
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            response = self._clean_response(response)

            return response if response else "¿En qué más puedo ayudarte?"

        except Exception as e:
            logger.error(f"Error generando respuesta: {e}")
            return "Lo siento, tuve un problema. ¿Podrías intentarlo de nuevo?"

    def _create_prompt(self, user_input: str, products_context: str) -> str:
        """Crea el prompt para el modelo"""
        # Historial de conversación
        history = ""
        for conv in self.conversation_history[-self.max_history:]:
            history += f"Usuario: {conv['user']}{self.tokenizer.eos_token}"
            history += f"Asistente: {conv['assistant']}{self.tokenizer.eos_token}"

        # Prompt completo
        prompt = f"{history}Usuario: {user_input}{self.tokenizer.eos_token}"

        if "No encontré" not in products_context:
            prompt = f"Contexto: {products_context}\n{prompt}"

        prompt += "Asistente:"

        return prompt

    def _clean_response(self, response: str) -> str:
        """Limpia la respuesta generada"""
        # Remover texto no deseado
        response = re.sub(r'\d{10,}', '', response)  # Remover números largos
        response = re.sub(r'http\S+', '', response)  # Remover URLs
        response = re.sub(r'\s+', ' ', response).strip()  # Normalizar espacios

        # Cortar en el primer signo de puntuación fuerte
        for cutoff in ['.', '!', '?', '\n']:
            if cutoff in response:
                response = response.split(cutoff)[0] + cutoff
                break

        return response

    def chat(self, user_input: str) -> str:
        """Maneja la conversación"""
        try:
            # Extraer intención
            intent = self._extract_search_intent(user_input)

            # Buscar productos
            products = self._search_products(intent)

            # Formatear contexto
            products_context = self._format_products_context(products)

            # Generar respuesta
            response = self._generate_response(user_input, products_context)

            # Actualizar historial
            self.conversation_history.append({
                'user': user_input,
                'assistant': response,
                'products_found': len(products)
            })

            # Limitar historial
            if len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]

            return response

        except Exception as e:
            logger.error(f"Error en chat: {e}")
            return "Lo siento, ocurrió un error. ¿Podrías intentarlo de nuevo?"

    def clear_conversation(self):
        """Limpia el historial"""
        self.conversation_history = []

    def get_conversation_stats(self) -> Dict:
        """Estadísticas de la conversación"""
        if not self.conversation_history:
            return {"total_interactions": 0, "success_rate": "0%"}

        total = len(self.conversation_history)
        successful = sum(1 for conv in self.conversation_history if conv.get('products_found', 0) > 0)

        return {
            "total_interactions": total,
            "successful_searches": successful,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%"
        }