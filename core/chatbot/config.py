import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EmbeddingConfig:
    """Configuración para el sistema de embeddings"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 100
    similarity_threshold: float = 0.3
    max_results: int = 10
    embeddings_path: str = "data/embeddings/"
    index_file: str = "product_index.faiss"
    metadata_file: str = "product_metadata.json"
    embeddings_file: str = "product_embeddings.pkl"


@dataclass
class ChatbotConfig:
    """Configuración para el chatbot principal"""
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    lora_path: str = "models/chatbot_lora"
    max_history: int = 5
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: float = 0.9

    # Configuración de memoria para RTX 4070 8GB
    load_in_4bit: bool = True
    use_gradient_checkpointing: bool = True
    torch_dtype: str = "float16"


@dataclass
class LoRAConfig:
    """Configuración para LoRA fine-tuning"""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = None
    lora_dropout: float = 0.05
    bias: str = "none"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class TrainingConfig:
    """Configuración para entrenamiento"""
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    max_length: int = 512
    output_dir: str = "models/chatbot_lora"


@dataclass
class SearchConfig:
    """Configuración para búsqueda de productos"""
    # Palabras clave para categorías
    category_keywords: Dict[str, List[str]] = None

    # Marcas comunes
    common_brands: List[str] = None

    # Palabras clave para ofertas
    deal_keywords: List[str] = None

    def __post_init__(self):
        if self.category_keywords is None:
            self.category_keywords = {
                'celulares': ['celular', 'telefono', 'smartphone', 'movil', 'iphone', 'galaxy'],
                'computadores': ['computador', 'pc', 'laptop', 'portatil', 'notebook', 'macbook'],
                'televisores': ['tv', 'television', 'televisor', 'pantalla', 'smart tv'],
                'audífonos': ['audifonos', 'auriculares', 'headphones', 'airpods'],
                'tablets': ['tablet', 'ipad'],
                'electrodomésticos': ['nevera', 'lavadora', 'microondas', 'licuadora', 'aspiradora'],
                'gaming': ['gaming', 'gamer', 'juegos', 'consola', 'ps4', 'ps5', 'xbox'],
                'accesorios': ['cable', 'cargador', 'funda', 'protector', 'soporte']
            }

        if self.common_brands is None:
            self.common_brands = [
                'samsung', 'apple', 'huawei', 'xiaomi', 'lg', 'sony',
                'lenovo', 'hp', 'dell', 'asus', 'acer', 'nokia',
                'motorola', 'oneplus', 'oppo', 'vivo', 'honor',
                'msi', 'alienware', 'razer', 'corsair', 'logitech'
            ]

        if self.deal_keywords is None:
            self.deal_keywords = [
                'oferta', 'descuento', 'barato', 'precio', 'promocion',
                'rebaja', 'outlet', 'liquidacion', 'especial', 'ganga'
            ]


class ChatbotSettings:
    """Clase principal de configuración"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file

        # Configuraciones
        self.embedding = EmbeddingConfig()
        self.chatbot = ChatbotConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.search = SearchConfig()

        # Cargar desde archivo si existe
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def load_from_file(self, config_file: str):
        """Carga configuración desde archivo JSON"""
        import json
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Actualizar configuraciones
            for key, value in config_data.get('embedding', {}).items():
                if hasattr(self.embedding, key):
                    setattr(self.embedding, key, value)

            for key, value in config_data.get('chatbot', {}).items():
                if hasattr(self.chatbot, key):
                    setattr(self.chatbot, key, value)

            for key, value in config_data.get('lora', {}).items():
                if hasattr(self.lora, key):
                    setattr(self.lora, key, value)

            for key, value in config_data.get('training', {}).items():
                if hasattr(self.training, key):
                    setattr(self.training, key, value)

            for key, value in config_data.get('search', {}).items():
                if hasattr(self.search, key):
                    setattr(self.search, key, value)

        except Exception as e:
            print(f"Error cargando configuración: {e}")

    def save_to_file(self, config_file: str):
        """Guarda configuración actual a archivo JSON"""
        import json
        from dataclasses import asdict

        config_data = {
            'embedding': asdict(self.embedding),
            'chatbot': asdict(self.chatbot),
            'lora': asdict(self.lora),
            'training': asdict(self.training),
            'search': asdict(self.search)
        }

        os.makedirs(os.path.dirname(config_file), exist_ok=True)

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    def get_paths(self) -> Dict[str, str]:
        """Retorna todas las rutas importantes"""
        return {
            'embeddings_dir': self.embedding.embeddings_path,
            'model_dir': self.chatbot.lora_path,
            'training_output': self.training.output_dir,
            'conversations_dir': 'data/conversations',
            'benchmarks_dir': 'data/benchmarks',
            'logs_dir': 'logs'
        }

    def create_directories(self):
        """Crea todos los directorios necesarios"""
        paths = self.get_paths()
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

    def validate_config(self) -> List[str]:
        """Valida la configuración y retorna lista de errores"""
        errors = []

        # Validar configuración de embeddings
        if self.embedding.batch_size <= 0:
            errors.append("batch_size debe ser mayor a 0")

        if not (0 <= self.embedding.similarity_threshold <= 1):
            errors.append("similarity_threshold debe estar entre 0 y 1")

        # Validar configuración de chatbot
        if self.chatbot.max_history <= 0:
            errors.append("max_history debe ser mayor a 0")

        if not (0 < self.chatbot.temperature <= 2):
            errors.append("temperature debe estar entre 0 y 2")

        if not (0 < self.chatbot.top_p <= 1):
            errors.append("top_p debe estar entre 0 y 1")

        # Validar configuración de LoRA
        if self.lora.r <= 0:
            errors.append("LoRA r debe ser mayor a 0")

        if self.lora.lora_alpha <= 0:
            errors.append("lora_alpha debe ser mayor a 0")

        # Validar configuración de entrenamiento
        if self.training.epochs <= 0:
            errors.append("epochs debe ser mayor a 0")

        if self.training.batch_size <= 0:
            errors.append("training batch_size debe ser mayor a 0")

        if self.training.learning_rate <= 0:
            errors.append("learning_rate debe ser mayor a 0")

        return errors


# Instancia global de configuración
settings = ChatbotSettings()


# Funciones de utilidad
def get_device():
    """Obtiene el dispositivo disponible"""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_memory_requirements():
    """Estima requerimientos de memoria del modelo"""
    import torch

    if not torch.cuda.is_available():
        return {"error": "No GPU available"}

    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Estimaciones para Mistral-7B
    base_model_memory = 14  # GB en float16
    lora_memory = 0.5  # GB adicional para LoRA
    inference_overhead = 2  # GB para inference

    total_required = base_model_memory + lora_memory + inference_overhead

    return {
        "available_memory_gb": gpu_memory,
        "required_memory_gb": total_required,
        "can_run": gpu_memory >= total_required,
        "recommendation": "Use 4-bit quantization" if gpu_memory < total_required else "Full precision OK"
    }


def setup_logging(log_level: str = "INFO"):
    """Configura el sistema de logging"""
    import logging
    from datetime import datetime

    # Crear directorio de logs
    os.makedirs("logs", exist_ok=True)

    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configurar handlers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/chatbot_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return log_file


# Constantes útiles
SUPPORTED_LANGUAGES = ['es', 'en']
DEFAULT_SYSTEM_PROMPT = """Eres un asistente de ventas especializado en productos tecnológicos de Colombia. 
Eres amigable, conocedor y ayudas a los usuarios a encontrar los mejores productos.
Responde de manera conversacional y natural, mencionando precios, descuentos y características relevantes.
Si hay productos disponibles, recomiéndalos. Si no hay productos, sugiere alternativas o pide más detalles."""

PRICE_RANGES = {
    "bajo": (0, 200000),
    "medio": (200000, 800000),
    "alto": (800000, 2000000),
    "premium": (2000000, float('inf'))
}

COMMON_QUERIES = [
    "Hola",
    "¿Qué productos tienes?",
    "Busco un celular",
    "¿Hay ofertas?",
    "Productos Samsung",
    "Computador para gaming",
    "Audífonos baratos",
    "TV 55 pulgadas"
]