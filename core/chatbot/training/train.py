# core/chatbot/training/train.py
from core.chatbot.vector_store import ProductVectorStore
from core.mongo.MongoManager import MongoManager
import time


def train_chatbot():
    print("🚀 Entrenando chatbot con productos de MongoDB...")
    start_time = time.time()

    # Verificar que hay datos
    mongo = MongoManager()
    product_count = mongo.get_product_count()
    print(f"📦 Productos en MongoDB: {product_count}")

    if product_count == 0:
        print("❌ No hay productos en la base de datos")
        return False

    # Construir vector store
    vector_store = ProductVectorStore()
    vector_store.build_vector_store()

    end_time = time.time()
    print(f"✅ Entrenamiento completado en {(end_time - start_time):.2f} segundos")
    return True


if __name__ == "__main__":
    train_chatbot()