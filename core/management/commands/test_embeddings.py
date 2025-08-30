from django.core.management.base import BaseCommand
from core.chatbot.EmbeddingManager import EmbeddingManager
import logging
import time

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Prueba el sistema de embeddings con consultas personalizadas'

    def add_arguments(self, parser):
        parser.add_argument(
            '--query',
            type=str,
            help='Consulta específica para probar'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.4,
            help='Umbral de similitud (default: 0.4)'
        )
        parser.add_argument(
            '--top-k',
            type=int,
            default=5,
            help='Número máximo de resultados (default: 5)'
        )
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Modo interactivo de pruebas'
        )
        parser.add_argument(
            '--recreate',
            action='store_true',
            help='Recrear embeddings antes de probar'
        )

    def handle(self, *args, **options):
        self.stdout.write("🚀 Iniciando pruebas de embeddings...")

        # Inicializar EmbeddingManager
        embedding_manager = EmbeddingManager()

        # Recrear embeddings si se solicita
        if options['recreate']:
            self.stdout.write("🔄 Recreando embeddings...")
            if embedding_manager.create_embeddings_from_db():
                self.stdout.write("✅ Embeddings recreados correctamente")
            else:
                self.stdout.write("❌ Error recreando embeddings")
                return

        if options['interactive']:
            self._interactive_mode(embedding_manager, options['threshold'], options['top_k'])
        elif options['query']:
            self._test_single_query(embedding_manager, options['query'], options['threshold'], options['top_k'])
        else:
            self._run_comprehensive_tests(embedding_manager)

    def _test_single_query(self, embedding_manager, query, threshold, top_k):
        """Prueba una consulta específica"""
        self.stdout.write(f"\n🔍 Probando: '{query}' (threshold: {threshold})")

        start_time = time.time()
        results = embedding_manager.search_products(query, top_k=top_k, threshold=threshold)
        search_time = time.time() - start_time

        if not results:
            self.stdout.write("   ❌ No se encontraron productos")
            return

        self.stdout.write(f"   ✅ Encontrados {len(results)} productos en {search_time:.2f}s:")

        for i, product in enumerate(results, 1):
            self.stdout.write(f"\n   {i}. 📦 {product['name']}")
            self.stdout.write(f"      🏷️  Marca: {product.get('brand', 'N/A')}")
            self.stdout.write(f"      💰 Precio: ${product.get('price', 0):,.0f}")

            discount = product.get('discount_percent', '0%')
            if discount not in ['0%', '0', 'Sin descuento']:
                self.stdout.write(f"      ⭐ Descuento: {discount}")

            self.stdout.write(f"      📋 Categoría: {product.get('category', 'N/A')}")
            self.stdout.write(f"      🔢 Similaridad: {product.get('similarity_score', 0):.3f}")

            # Mostrar specs importantes
            specs = product.get('specifications', {})
            if specs:
                important_specs = []
                spec_keys = list(specs.keys())
                # Mostrar hasta 3 specs más relevantes
                for key in spec_keys[:3]:
                    important_specs.append(f"{key}: {specs[key]}")
                if important_specs:
                    self.stdout.write(f"      ⚙️  Specs: {', '.join(important_specs)}")

    def _interactive_mode(self, embedding_manager, threshold, top_k):
        """Modo interactivo para probar múltiples consultas"""
        self.stdout.write(
            "\n💻 MODO INTERACTIVO - Escribe consultas para probar\n"
            "   Comandos:\n"
            "   - 'threshold X' → Cambiar umbral (ej: threshold 0.4)\n"
            "   - 'topk X' → Cambiar top-k (ej: topk 3)\n"
            "   - 'exit' → Salir\n"
            "   - 'stats' → Ver estadísticas\n"
            "   - 'recreate' → Recrear embeddings\n"
        )

        current_threshold = threshold
        current_top_k = top_k

        while True:
            try:
                query = input("\n🔎 Consulta: ").strip()

                if query.lower() in ['exit', 'quit', 'salir']:
                    break
                elif query.lower() == 'stats':
                    stats = embedding_manager.get_stats()
                    self._show_stats(stats)
                    continue
                elif query.lower() == 'recreate':
                    self.stdout.write("🔄 Recreando embeddings...")
                    if embedding_manager.create_embeddings_from_db():
                        self.stdout.write("✅ Embeddings recreados correctamente")
                    else:
                        self.stdout.write("❌ Error recreando embeddings")
                    continue
                elif query.lower().startswith('threshold '):
                    try:
                        new_threshold = float(query.split()[1])
                        current_threshold = new_threshold
                        self.stdout.write(f"   ✅ Nuevo threshold: {current_threshold}")
                        continue
                    except:
                        self.stdout.write("   ❌ Formato: threshold 0.4")
                        continue
                elif query.lower().startswith('topk '):
                    try:
                        new_topk = int(query.split()[1])
                        current_top_k = new_topk
                        self.stdout.write(f"   ✅ Nuevo top-k: {current_top_k}")
                        continue
                    except:
                        self.stdout.write("   ❌ Formato: topk 5")
                        continue

                if not query:
                    continue

                self._test_single_query(embedding_manager, query, current_threshold, current_top_k)

            except KeyboardInterrupt:
                self.stdout.write("\n\n🛑 Pruebas interrumpidas")
                break
            except Exception as e:
                self.stdout.write(f"   ❌ Error: {e}")

    def _run_comprehensive_tests(self, embedding_manager):
        """Ejecuta una batería completa de pruebas"""
        test_cases = [
            ("computador portátil gamer victus", 0.35),
            ("celular samsung", 0.35),
            ("smartphone motorola", 0.35),
            ("tablet android", 0.35),
            ("laptop i7 16gb ram", 0.3),
            ("iphone 128gb", 0.3),
            ("televisor smart 55", 0.35),
            ("audífonos gaming", 0.35),
            ("monitor 24 pulgadas", 0.35),
        ]

        self.stdout.write("🧪 Ejecutando batería de pruebas...\n")

        for query, threshold in test_cases:
            self.stdout.write(f"🧪 TEST: '{query}'")
            self._test_single_query(embedding_manager, query, threshold, 3)
            self.stdout.write("   " + "─" * 60)

        # Mostrar estadísticas finales
        stats = embedding_manager.get_stats()
        self._show_stats(stats)

    def _show_stats(self, stats):
        """Muestra estadísticas del índice"""
        self.stdout.write("\n📊 ESTADÍSTICAS DEL ÍNDICE:")
        self.stdout.write(f"   📦 Total productos: {stats['total_products']}")

        if stats['top_brands']:
            self.stdout.write(f"   🏷️  Marcas principales: {', '.join(list(stats['top_brands'].keys())[:5])}")

        if stats['categories']:
            self.stdout.write(f"   📋 Categorías: {', '.join(list(stats['categories'].keys())[:5])}")

        self.stdout.write(
            f"   💰 Productos con descuento: {stats['products_with_discount']} ({stats['discount_percentage']})")

        if stats['price_ranges']:
            self.stdout.write("   💵 Distribución de precios:")
            for range_name, count in stats['price_ranges'].items():
                percentage = (count / stats['total_products'] * 100) if stats['total_products'] > 0 else 0
                self.stdout.write(f"      {range_name}: {count} productos ({percentage:.1f}%)")