import pymongo
from pymongo import MongoClient, UpdateOne
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoManager:
    def __init__(self, connection_string: str = None, db_name: str = "alkosto_db"):
        """
        Inicializa el manager de MongoDB

        Args:
            connection_string: String de conexión a MongoDB
            db_name: Nombre de la base de datos
        """
        self.connection_string = connection_string or "mongodb://localhost:27017/"
        self.db_name = db_name
        self.client = None
        self.db = None
        self.products_collection = None

        self.connect()

    def connect(self):
        """Establece conexión con MongoDB"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )

            # Verificar conexión
            self.client.admin.command('ping')
            logger.info("✅ Conexión exitosa a MongoDB")

            self.db = self.client[self.db_name]
            self.products_collection = self.db['products']

            # Crear índices para optimizar búsquedas
            self._create_indexes()

        except pymongo.errors.ServerSelectionTimeoutError:
            logger.error("❌ No se pudo conectar a MongoDB")
            raise
        except Exception as e:
            logger.error(f"❌ Error de conexión: {e}")
            raise

    def _create_indexes(self):
        """Crea índices para optimizar las consultas"""
        indexes = [
            [("name", pymongo.TEXT)],  # Índice de texto para búsquedas
            [("category", 1)],  # Índice por categoría
            [("brand", 1)],  # Índice por marca
            [("discount_percent", -1)],  # Índice por descuento (descendente)
            [("scraping_date", -1)],  # Índice por fecha de scraping
            [("product_url", 1)],  # Índice único para URLs
        ]

        for index in indexes:
            try:
                self.products_collection.create_index(index)
            except Exception as e:
                logger.warning(f"⚠️ Error creando índice: {e}")

    def save_products(self, products: List[Dict[str, Any]], category: str = None):
        """
        Guarda productos en MongoDB, actualizando existentes

        Args:
            products: Lista de diccionarios con datos de productos
            category: Categoría de los productos (opcional)
        """
        if not products:
            logger.warning("⚠️ No hay productos para guardar")
            return 0

        try:
            operations = []
            saved_count = 0
            updated_count = 0

            for product in products:
                # Añadir timestamp y categoría si se proporciona
                product['last_updated'] = datetime.now()
                if category and 'category' not in product:
                    product['category'] = category

                # Crear operación de upsert (insertar o actualizar)
                operation = UpdateOne(
                    {'product_url': product['product_url']},  # Filtro por URL única
                    {'$set': product},  # Datos a actualizar
                    upsert=True  # Insertar si no existe
                )
                operations.append(operation)

            # Ejecutar operaciones en lote
            if operations:
                result = self.products_collection.bulk_write(operations)
                saved_count = result.upserted_count
                updated_count = result.modified_count

                logger.info(f"💾 Guardados: {saved_count} nuevos, Actualizados: {updated_count} productos")

            return saved_count + updated_count

        except Exception as e:
            logger.error(f"❌ Error guardando productos: {e}")
            return 0

    def get_product_by_url(self, product_url: str):
        """Obtiene un producto por su URL"""
        try:
            return self.products_collection.find_one({'product_url': product_url})
        except Exception as e:
            logger.error(f"❌ Error obteniendo producto: {e}")
            return None

    def get_products_by_category(self, category: str, limit: int = 100):
        """Obtiene productos por categoría"""
        try:
            return list(self.products_collection.find(
                {'category': category}
            ).sort('scraping_date', -1).limit(limit))
        except Exception as e:
            logger.error(f"❌ Error obteniendo productos por categoría: {e}")
            return []

    def get_products_with_discount(self, min_discount: float = 10, limit: int = 50):
        """Obtiene productos con descuento mínimo"""
        try:
            return list(self.products_collection.find({
                'discount_percent': {'$ne': "0%"},
                'discount_price_num': {'$gt': 0},
                'original_price_num': {'$gt': 0},
                '$expr': {
                    '$gte': [
                        {'$subtract': ['$original_price_num', '$discount_price_num']},
                        min_discount
                    ]
                }
            }).sort('discount_percent', -1).limit(limit))
        except Exception as e:
            logger.error(f"❌ Error obteniendo productos con descuento: {e}")
            return []

    def search_products(self, search_term: str, limit: int = 50):
        """Busca productos por texto"""
        try:
            return list(self.products_collection.find(
                {'$text': {'$search': search_term}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit))
        except Exception as e:
            logger.error(f"❌ Error buscando productos: {e}")
            return []

    def get_product_count(self):
        """Obtiene el número total de productos"""
        try:
            return self.products_collection.count_documents({})
        except Exception as e:
            logger.error(f"❌ Error contando productos: {e}")
            return 0

    def get_categories(self):
        """Obtiene lista de categorías únicas"""
        try:
            return self.products_collection.distinct('category')
        except Exception as e:
            logger.error(f"❌ Error obteniendo categorías: {e}")
            return []

    def delete_old_products(self, days_old: int = 30):
        """Elimina productos más viejos que X días"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            result = self.products_collection.delete_many({
                'scraping_date': {'$lt': cutoff_date.isoformat()}
            })
            logger.info(f"🗑️ Eliminados {result.deleted_count} productos viejos")
            return result.deleted_count
        except Exception as e:
            logger.error(f"❌ Error eliminando productos viejos: {e}")
            return 0

    def close_connection(self):
        """Cierra la conexión con MongoDB"""
        if self.client:
            self.client.close()
            logger.info("🔌 Conexión a MongoDB cerrada")

    def __enter__(self):
        """Para usar con context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cierra conexión al salir del context manager"""
        self.close_connection()