from django.core.management.base import BaseCommand
from core.chatbot.EmbeddingManager import EmbeddingManager
import logging

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
            default=0.3,
            help='Umbral de similitud (default: 0.3)'
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

    def handle(self, *args, **options):
        self.stdout.write("🚀 Iniciando pruebas de embeddings...")

        # Inicializar EmbeddingManager
        embedding_manager = EmbeddingManager()

        if options['interactive']:
            self._interactive_mode(embedding_manager, options['threshold'], options['top_k'])
        elif options['query']:
            self._test_single_query(embedding_manager, options['query'], options['threshold'], options['top_k'])
        else:
            self._run_comprehensive_tests(embedding_manager)

    def _test_single_query(self, embedding_manager, query, threshold, top_k):
        """Prueba una consulta específica"""
        self.stdout.write(f"\n🔍 Probando: '{query}' (threshold: {threshold})")

        results = embedding_manager.search_products(query, top_k=top_k, threshold=threshold)

        if not results:
            self.stdout.write("   ❌ No se encontraron productos")
            return

        self.stdout.write(f"   ✅ Encontrados {len(results)} productos:")

        for i, product in enumerate(results, 1):
            self.stdout.write(f"\n   {i}. 📦 {product['name']}")
            self.stdout.write(f"      🏷️  Marca: {product.get('brand', 'N/A')}")
            self.stdout.write(f"      💰 Precio: ${product.get('price', 0):,.0f}")

            if product.get('discount_percent') not in [None, '0%', '0']:
                self.stdout.write(f"      ⭐ Descuento: {product.get('discount_percent')}")

            self.stdout.write(f"      📋 Categoría: {product.get('category', 'N/A')}")
            self.stdout.write(f"      🔢 Similaridad: {product.get('similarity_score', 0):.3f}")

            # Mostrar specs importantes
            specs = product.get('specifications', {})
            if specs:
                important_specs = []
                for key in ['RAM', 'Almacenamiento', 'Procesador', 'Pantalla', 'Memoria']:
                    if key in specs:
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
            ("computador portátil", 0.3),
            ("celular samsung", 0.3),
            ("televisor smart", 0.3),
            ("tablet", 0.3),
            ("laptop i7 16gb", 0.25),
            ("iphone 128gb", 0.25),
        ]

        self.stdout.write("🧪 Ejecutando batería de pruebas...")

        for query, threshold in test_cases:
            self._test_single_query(embedding_manager, query, threshold, 3)
            self.stdout.write("   " + "─" * 60)

        # Mostrar estadísticas finales
        stats = embedding_manager.get_stats()
        self._show_stats(stats)

    def _show_stats(self, stats):
        """Muestra estadísticas del índice"""
        self.stdout.write("\n📊 ESTADÍSTICAS DEL ÍNDICE:")
        self.stdout.write(f"   📦 Total productos: {stats['total_products']}")
        self.stdout.write(f"   🏷️  Marcas principales: {', '.join(list(stats['top_brands'].keys())[:5])}")
        self.stdout.write(f"   📋 Categorías: {', '.join(list(stats['categories'].keys())[:5])}")
        self.stdout.write(
            f"   💰 Productos con descuento: {stats['products_with_discount']} ({stats['discount_percentage']})")