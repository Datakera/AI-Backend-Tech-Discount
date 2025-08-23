# core/chatbot/vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.mongo.MongoManager import MongoManager
import os


class ProductVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store = None
        self.persist_dir = "./data/chroma_db"

    def create_documents_from_mongo(self):
        """Crea documentos LangChain desde MongoDB"""
        mongo = MongoManager()
        products = mongo.get_all_products()

        documents = []
        for product in products:
            # Crear texto enriquecido para embeddings
            text = f"""
            Producto: {product['name']}
            Marca: {product['brand']}
            Categoría: {product['category']}
            Precio original: {product['original_price']}
            Precio con descuento: {product['discount_price']}
            Descuento: {product['discount_percent']}
            Especificaciones: {', '.join([f'{k}: {v}' for k, v in product['specifications'].items()])}
            Rating: {product['rating']}
            """

            metadata = {
                "product_id": str(product["_id"]),
                "name": product["name"],
                "brand": product["brand"],
                "category": product["category"],
                "price": product["discount_price_num"],
                "discount": product["discount_percent"],
                "url": product["product_url"]
            }

            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def build_vector_store(self):
        """Construye y persiste el vector store"""
        documents = self.create_documents_from_mongo()

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        self.vector_store.persist()
        print(f"✅ Vector store creado con {len(documents)} productos")

    def load_vector_store(self):
        """Carga el vector store existente"""
        if os.path.exists(self.persist_dir):
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            return True
        return False

    def search_similar_products(self, query, k=5):
        """Busca productos similares a la consulta"""
        if not self.vector_store:
            if not self.load_vector_store():
                self.build_vector_store()

        return self.vector_store.similarity_search(query, k=k)