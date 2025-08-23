# core/chatbot/chains/product_chain.py
from langchain.chains.base import Chain
from typing import Dict, List, Any
from core.mongo.MongoManager import MongoManager


class ProductSpecificChain(Chain):
    """Cadena especializada para búsqueda específica de productos"""

    @property
    def input_keys(self) -> List[str]:
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        return ["result", "products"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs["query"]
        mongo = MongoManager()

        # Búsqueda específica por características
        if "ram" in query.lower():
            products = mongo.search_products_by_spec("Memoria RAM", "16GB")
        elif "ssd" in query.lower():
            products = mongo.search_products_by_spec("Almacenamiento", "SSD")
        else:
            products = []

        return {
            "result": f"Encontré {len(products)} productos que coinciden",
            "products": products
        }