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
    """Entrenador del chatbot con LoRA fine-tuning"""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuración de LoRA
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Rutas de guardado
        self.output_dir = "models/chatbot_lora"
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"💻 Dispositivo: {self.device}")

    def load_model(self, load_in_4bit: bool = True):
        """Carga el modelo base con configuración para RTX 4070 8GB - VERSIÓN CORREGIDA"""
        try:
            logger.info(f"🔄 Cargando modelo {self.model_name}...")

            # Configuración para 4-bit quantization
            if load_in_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",  # ✅ Deja que Hugging Face maneje el device mapping
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None,  # ✅ Auto solo si hay GPU
                    trust_remote_code=True
                )

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Configurar pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Aplicar LoRA
            self.model = get_peft_model(self.model, self.lora_config)

            # Mostrar parámetros entrenables
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            logger.info(f"✅ Modelo cargado correctamente")
            logger.info(
                f"📊 Parámetros entrenables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    def create_training_dataset(self) -> Dataset:
        """Crea dataset de entrenamiento desde la base de datos"""
        try:
            logger.info("📊 Creando dataset de entrenamiento...")

            # Obtener productos de la base de datos
            mongo = MongoManager()
            products = mongo.get_all_products(limit=2000)  # Limitar para entrenamiento

            if not products:
                raise ValueError("No hay productos en la base de datos")

            # Crear conversaciones de ejemplo
            conversations = []

            for product in products:
                conversations.extend(self._create_product_conversations(product))

            # Crear ejemplos generales
            conversations.extend(self._create_general_conversations())

            # Formatear para entrenamiento
            formatted_data = []
            for conv in conversations:
                formatted_text = self._format_conversation(conv)
                formatted_data.append({"text": formatted_text})

            logger.info(f"📝 Dataset creado con {len(formatted_data)} ejemplos")

            return Dataset.from_list(formatted_data)

        except Exception as e:
            logger.error(f"❌ Error creando dataset: {e}")
            raise

    def _create_product_conversations(self, product: Dict) -> List[Dict]:
        """Crea conversaciones de ejemplo para un producto"""
        name = product.get('name', 'Producto')
        brand = product.get('brand', 'Sin marca')
        category = product.get('category', 'Sin categoría')
        price = product.get('discount_price_num', product.get('original_price_num', 0))
        discount = product.get('discount_percent', '0%')
        url = product.get('product_url', '')

        conversations = []

        # Conversación 1: Búsqueda directa
        conversations.append({
            "user": f"Busco {category.lower()}",
            "assistant": f"¡Perfecto! Te recomiendo el {name} de {brand}. Está disponible por ${price:,.0f}. " +
                         (f"¡Tiene {discount} de descuento! " if discount != '0%' else "") +
                         f"¿Te interesa conocer más detalles de este producto?"
        })

        # Conversación 2: Precio específico
        if discount != '0%':
            conversations.append({
                "user": f"¿Hay ofertas en {category.lower()}?",
                "assistant": f"¡Sí! El {name} de {brand} tiene {discount} de descuento. " +
                             f"Su precio actual es ${price:,.0f}. Es una excelente oportunidad."
            })

        # Conversación 3: Marca específica
        if brand != 'Sin marca':
            conversations.append({
                "user": f"¿Tienes productos de {brand}?",
                "assistant": f"¡Por supuesto! Tengo el {name} de {brand} disponible por ${price:,.0f}. " +
                             f"Es un excelente {category.lower()}. ¿Te gustaría más información?"
            })

        return conversations

    def _create_general_conversations(self) -> List[Dict]:
        """Crea conversaciones generales del chatbot"""
        return [
            {
                "user": "Hola",
                "assistant": "¡Hola! Soy tu asistente de productos tecnológicos. Te ayudo a encontrar los mejores productos y ofertas en tiendas de Colombia. ¿Qué estás buscando hoy?"
            },
            {
                "user": "¿Qué productos tienes?",
                "assistant": "Tengo una amplia variedad de productos tecnológicos: celulares, computadores, tablets, audífonos, televisores, electrodomésticos y mucho más. Todos con precios actualizados y ofertas especiales. ¿Hay algo específico que te interese?"
            },
            {
                "user": "¿Hay ofertas?",
                "assistant": "¡Claro que sí! Constantemente actualizo las mejores ofertas y descuentos disponibles. ¿En qué categoría de productos te gustaría ver ofertas? Puedo mostrarte celulares, computadores, electrodomésticos o cualquier otra categoría."
            },
            {
                "user": "¿Cómo funciona esto?",
                "assistant": "Es muy fácil: solo dime qué producto buscas, tu presupuesto o si quieres ver ofertas especiales. Yo busco en mi base de datos actualizada y te muestro las mejores opciones disponibles en tiendas de Colombia. ¡Empecemos!"
            },
            {
                "user": "¿Los precios están actualizados?",
                "assistant": "Sí, los precios se actualizan regularmente mediante scraping de las principales tiendas de Colombia. Siempre te muestro la información más reciente disponible, incluyendo descuentos y ofertas especiales."
            }
        ]

    def _format_conversation(self, conversation: Dict) -> str:
        """Formatea una conversación para el entrenamiento"""
        user_msg = conversation["user"]
        assistant_msg = conversation["assistant"]

        # Formato para Mistral Instruct
        return f"<s>[INST] {user_msg} [/INST] {assistant_msg}</s>"

    def tokenize_function(self, examples):
        """Tokeniza los ejemplos para entrenamiento - VERSIÓN CORREGIDA"""
        # Tokenizar sin return_tensors para que datasets pueda manejar los arrays
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,  # ✅ Cambiado a True
            max_length=512,
            # ❌ REMOVED: return_tensors="pt" - Esto causa el error
        )

        # Asegurar que todas las secuencias tengan la misma longitud
        # agregando padding donde sea necesario
        return tokenized

    def train(self, dataset: Dataset, epochs: int = 3, batch_size: int = 4):
        """Entrena el modelo con LoRA - VERSIÓN CORREGIDA"""
        try:
            logger.info("🚀 Iniciando entrenamiento...")

            # Tokenizar dataset (sin return_tensors)
            tokenized_dataset = dataset.map(self.tokenize_function, batched=True)

            # Configuración de entrenamiento
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_steps=500,
                save_total_limit=2,
                remove_unused_columns=True,  # ✅ Cambiado a True
                dataloader_drop_last=True,
                gradient_checkpointing=True,
            )

            # Data collator con padding dinámico
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # ✅ Para mejor rendimiento en GPU
            )

            # Crear trainer
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
            logger.error(f"❌ Error durante entrenamiento: {e}")
            raise

    def save_training_config(self, config: Dict):
        """Guarda la configuración del entrenamiento"""
        config_file = os.path.join(self.output_dir, "training_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"💾 Configuración guardada en: {config_file}")
