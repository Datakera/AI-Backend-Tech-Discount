import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging
from typing import List, Dict
from core.mongo.MongoManager import MongoManager

logger = logging.getLogger(__name__)


class ChatbotTrainer:
    """Entrenador optimizado para chatbot de productos tecnológicos"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuración de LoRA optimizada
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj"],  # Para DialoGPT/GPT-2
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Rutas de guardado
        self.output_dir = "models/tech_chatbot"
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Modelo: {self.model_name}")

    def load_model(self):
        """Carga el modelo y tokenizer"""
        try:
            logger.info(f"Cargando modelo {self.model_name}...")

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Configurar tokens especiales para DialoGPT
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )

            # Aplicar LoRA
            self.model = get_peft_model(self.model, self.lora_config)

            # Mover a dispositivo si es necesario
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            # Mostrar estadísticas
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            logger.info("Modelo cargado correctamente")
            logger.info(f"Parámetros entrenables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise

    def create_training_dataset(self, max_samples: int = 5000) -> Dataset:
        """Crea dataset de entrenamiento desde MongoDB"""
        try:
            logger.info("Creando dataset de entrenamiento...")

            # Obtener productos de MongoDB
            mongo = MongoManager()
            products = mongo.get_all_products(limit=10000)  # Obtener más para filtrar

            if not products or not isinstance(products, list):
                logger.warning("No se obtuvieron productos válidos")
                return self._create_fallback_dataset()

            # Filtrar productos válidos
            valid_products = []
            for product in products:
                if isinstance(product, dict) and product.get('name') and product.get('brand'):
                    # Convertir precios a números
                    try:
                        discount_price = product.get('discount_price_num', 0)
                        if isinstance(discount_price, str):
                            discount_price = float(discount_price.replace(',', '').replace('$', '').strip())
                        product['discount_price_num'] = discount_price

                        original_price = product.get('original_price_num', 0)
                        if isinstance(original_price, str):
                            original_price = float(original_price.replace(',', '').replace('$', '').strip())
                        product['original_price_num'] = original_price

                        valid_products.append(product)
                    except (ValueError, AttributeError):
                        continue

            logger.info(f"Productos válidos obtenidos: {len(valid_products)}")

            # Crear conversaciones de entrenamiento
            conversations = []
            for product in valid_products[:max_samples]:
                conversations.extend(self._create_product_conversations(product))

            # Agregar conversaciones generales
            conversations.extend(self._create_general_conversations())

            # Limitar y mezclar
            conversations = conversations[:max_samples]

            # Formatear para el modelo
            formatted_data = []
            for conv in conversations:
                formatted_text = self._format_conversation(conv)
                formatted_data.append({"text": formatted_text})

            logger.info(f"Dataset creado con {len(formatted_data)} ejemplos")
            return Dataset.from_list(formatted_data)

        except Exception as e:
            logger.error(f"Error creando dataset: {e}")
            return self._create_fallback_dataset()

    def _create_product_conversations(self, product: Dict) -> List[Dict]:
        """Crea conversaciones para un producto"""
        conversations = []

        name = product.get('name', '')[:100]
        brand = product.get('brand', '')
        price = product.get('discount_price_num') or product.get('original_price_num', 0)
        discount = product.get('discount_percent', '0%')
        category = product.get('category', '')

        if not name or price <= 0:
            return conversations

        # Conversaciones básicas
        base_conversations = [
            {
                "user": f"ofertas {brand} {category}",
                "assistant": f"Tenemos {name} de {brand} por ${price:,.0f}. {f'¡En oferta con {discount} de descuento!' if discount != '0%' else ''}"
            },
            {
                "user": f"precio de {name}",
                "assistant": f"El {name} tiene un precio de ${price:,.0f}{f' con {discount} de descuento' if discount != '0%' else ''}. ¡Es una excelente opción!"
            },
            {
                "user": f"{brand} {category}",
                "assistant": f"Te recomiendo el {name} de {brand} por ${price:,.0f}. {f'Tiene {discount} de descuento.' if discount != '0%' else 'Es una gran relación calidad-precio.'}"
            }
        ]

        return base_conversations

    def _create_general_conversations(self) -> List[Dict]:
        """Conversaciones generales del chatbot"""
        return [
            {"user": "hola",
             "assistant": "¡Hola! Soy tu asistente de productos tecnológicos. ¿En qué puedo ayudarte hoy?"},
            {"user": "qué productos tienes",
             "assistant": "Tenemos celulares, computadores, tablets, televisores, audífonos y muchos más productos tecnológicos con las mejores ofertas."},
            {"user": "ofertas",
             "assistant": "¡Tenemos excelentes ofertas! ¿Qué tipo de producto te interesa? Puedo ayudarte a encontrar las mejores promociones."},
            {"user": "gracias",
             "assistant": "¡De nada! Estoy aquí para ayudarte. ¿Hay algo más en lo que pueda asistirte?"},
            {"user": "ayuda",
             "assistant": "Puedo ayudarte a: 1) Buscar productos por categoría 2) Encontrar ofertas 3) Recomendarte productos según tu necesidad. ¿Por dónde empezamos?"}
        ]

    def _format_conversation(self, conversation: Dict) -> str:
        """Formatea conversación para DialoGPT"""
        # Usar el token EOS de DialoGPT directamente si el tokenizer no está cargado
        eos_token = " <|endoftext|> " if self.tokenizer is None else f" {self.tokenizer.eos_token} "
        return f"{conversation['user']}{eos_token}{conversation['assistant']}{eos_token}"

    def tokenize_function(self, examples):
        """Tokenización para el dataset"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_overflowing_tokens=False,
        )

    def train(self, dataset: Dataset, epochs: int = 3, batch_size: int = 4):
        """Entrena el modelo"""
        try:
            logger.info("Iniciando entrenamiento...")

            # Tokenizar dataset
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

            # Configuración de entrenamiento
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=100,
                learning_rate=3e-4,
                logging_steps=50,
                save_steps=500,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                remove_unused_columns=True,
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # Entrenar
            trainer.train()

            # Guardar modelo
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)

            logger.info(f"✅ Entrenamiento completado. Modelo guardado en: {self.output_dir}")

        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            raise