from django.core.management.base import BaseCommand
from core.chatbot.vector_store import ProductVectorStore
from core.mongo.MongoManager import MongoManager


class Command(BaseCommand):
    help = 'Debug del chatbot - Verifica cada componente'

    def handle(self, *args, **options):
        self.stdout.write("🔍 Iniciando debug del chatbot...")

        # 1. Verificar MongoDB
        self.stdout.write("\n1. 📊 Verificando MongoDB...")
        try:
            mongo = MongoManager()
            count = mongo.get_product_count()
            self.stdout.write(self.style.SUCCESS(f"   ✅ MongoDB conectado: {count} productos"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ❌ MongoDB error: {e}"))
            return

        # 2. Verificar Vector Store
        self.stdout.write("\n2. 🗄️ Verificando Vector Store...")
        try:
            vector_store = ProductVectorStore()
            if vector_store.load_vector_store():
                self.stdout.write(self.style.SUCCESS("   ✅ Vector Store cargado"))
            else:
                self.stdout.write(self.style.WARNING("   ⚠️ Vector Store no existe, construyendo..."))
                vector_store.build_vector_store()
                self.stdout.write(self.style.SUCCESS("   ✅ Vector Store construido"))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ❌ Vector Store error: {e}"))
            return

        # 3. Probar búsqueda
        self.stdout.write("\n3. 🔍 Probando búsqueda...")
        try:
            results = vector_store.search_similar_products("laptop 16GB RAM", k=3)
            self.stdout.write(self.style.SUCCESS(f"   ✅ Búsqueda exitosa: {len(results)} resultados"))

            for i, result in enumerate(results, 1):
                self.stdout.write(f"      {i}. {result.metadata.get('name', 'Sin nombre')[:50]}...")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"   ❌ Búsqueda error: {e}"))
            return

        self.stdout.write(self.style.SUCCESS("\n🎉 Todos los componentes funcionan correctamente!"))