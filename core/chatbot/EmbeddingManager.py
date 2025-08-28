import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import logging
from core.mongo.MongoManager import MongoManager

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Maneja la creación y búsqueda de embeddings para productos"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.product_metadata = []
        self.embeddings_path = "data/embeddings/"

        # Crear directorio si no existe
        os.makedirs(self.embeddings_path, exist_ok=True)

        self.index_file = os.path.join(self.embeddings_path, "product_index.faiss")
        self.metadata_file = os.path.join(self.embeddings_path, "product_metadata.json")
        self.embeddings_file = os.path.join(self.embeddings_path, "product_embeddings.pkl")

        self._load_model()
        self._load_or_create_index()

    def _load_model(self):
        """Carga el modelo de sentence transformers"""
        try:
            logger.info(f"Cargando modelo {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Modelo de embeddings cargado correctamente")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def _create_product_text(self, product: Dict) -> str:
        """Crea texto completo para embeddings"""
        text_parts = [
            f"Nombre: {product.get('name', '')}",
            f"Marca: {product.get('brand', '')}",
            f"Categoría: {product.get('category', '')}",
            f"Precio: {product.get('discount_price_num', product.get('original_price_num', 0))}",
            f"Descuento: {product.get('discount_percent', '0%')}",
        ]

        # Incluir todas las especificaciones
        specs = product.get('specifications', {})
        for key, value in specs.items():
            text_parts.append(f"{key}: {value}")

        return " | ".join(text_parts)

    def create_embeddings_from_db(self, batch_size: int = 100) -> bool:
        """Crea embeddings para todos los productos en la base de datos"""
        try:
            logger.info("🔄 Iniciando creación de embeddings...")

            # Obtener productos de MongoDB
            mongo = MongoManager()
            products = mongo.get_all_products()

            if not products:
                logger.warning("⚠️ No hay productos en la base de datos")
                return False

            logger.info(f"📦 Procesando {len(products)} productos...")

            # Crear textos para embedding
            product_texts = []
            metadata = []

            for product in products:
                text = self._create_product_text(product)
                product_texts.append(text)

                # Guardar metadata importante
                metadata.append({
                    'id': str(product.get('_id')),
                    'name': product.get('name', ''),
                    'brand': product.get('brand', ''),
                    'category': product.get('category', ''),
                    'price': product.get('discount_price_num', product.get('original_price_num', 0)),
                    'discount_percent': product.get('discount_percent', '0%'),
                    'product_url': product.get('product_url', ''),
                    'image_url': product.get('image_url', ''),
                    'availability': product.get('availability', 'Disponible'),
                    'specifications': product.get('specifications', {})
                })

            # Crear embeddings en lotes
            all_embeddings = []
            for i in range(0, len(product_texts), batch_size):
                batch = product_texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=True)
                all_embeddings.append(batch_embeddings)

            # Concatenar todos los embeddings
            embeddings = np.vstack(all_embeddings)

            # Crear índice FAISS
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product para similitud

            # Normalizar embeddings para usar cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            # Guardar índice y metadata
            faiss.write_index(self.index, self.index_file)

            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)

            self.product_metadata = metadata

            logger.info(f"✅ Embeddings creados correctamente: {embeddings.shape}")
            logger.info(f"💾 Índice guardado en: {self.index_file}")

            return True

        except Exception as e:
            logger.error(f"❌ Error creando embeddings: {e}")
            return False

    def _load_or_create_index(self):
        """Carga el índice existente o solicita crearlo"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                logger.info("📂 Cargando índice existente...")
                self.index = faiss.read_index(self.index_file)

                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.product_metadata = json.load(f)

                logger.info(f"✅ Índice cargado: {self.index.ntotal} productos")
            else:
                logger.info("⚠️ No se encontró índice existente. Use create_embeddings_from_db() para crearlo")

        except Exception as e:
            logger.error(f"❌ Error cargando índice: {e}")

    def search_products(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[Dict]:
        """Busca productos similares a la consulta"""
        try:
            if self.index is None or not self.product_metadata:
                logger.error("❌ Índice no cargado. Ejecute create_embeddings_from_db() primero")
                return []

            # Crear embedding de la consulta
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)

            # Buscar productos similares
            scores, indices = self.index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold:  # Filtrar por umbral de similitud
                    product = self.product_metadata[idx].copy()
                    product['similarity_score'] = float(score)
                    results.append(product)

            logger.info(f"🔍 Encontrados {len(results)} productos para: '{query}'")
            return results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}")
            return []

    def search_by_category_and_price(self, category: str = None,
                                     min_price: float = None, max_price: float = None,
                                     with_discount: bool = False, top_k: int = 10) -> List[Dict]:
        """Búsqueda avanzada por filtros"""
        try:
            filtered_results = []

            for i, product in enumerate(self.product_metadata):
                # Filtrar por categoría
                if category and category.lower() not in product['category'].lower():
                    continue

                # Filtrar por precio
                price = product['price']
                if min_price is not None and price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue

                # Filtrar por descuento
                if with_discount and product['discount_percent'] == '0%':
                    continue

                filtered_results.append(product)

                if len(filtered_results) >= top_k:
                    break

            return filtered_results

        except Exception as e:
            logger.error(f"❌ Error en búsqueda avanzada: {e}")
            return []

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del índice"""
        if not self.product_metadata:
            return {}

        categories = {}
        brands = {}
        price_ranges = {"0-100k": 0, "100k-500k": 0, "500k-1M": 0, "1M+": 0}
        with_discount = 0

        for product in self.product_metadata:
            # Contar categorías
            cat = product['category']
            categories[cat] = categories.get(cat, 0) + 1

            # Contar marcas
            brand = product['brand']
            brands[brand] = brands.get(brand, 0) + 1

            # Rangos de precio
            price = product['price']
            if price < 100000:
                price_ranges["0-100k"] += 1
            elif price < 500000:
                price_ranges["100k-500k"] += 1
            elif price < 1000000:
                price_ranges["500k-1M"] += 1
            else:
                price_ranges["1M+"] += 1

            # Productos con descuento
            if product['discount_percent'] != '0%':
                with_discount += 1

        return {
            'total_products': len(self.product_metadata),
            'categories': dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_brands': dict(sorted(brands.items(), key=lambda x: x[1], reverse=True)[:10]),
            'price_ranges': price_ranges,
            'products_with_discount': with_discount,
            'discount_percentage': f"{(with_discount / len(self.product_metadata) * 100):.1f}%"
        }
