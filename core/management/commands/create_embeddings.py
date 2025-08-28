from django.core.management.base import BaseCommand
from core.chatbot.EmbeddingManager import EmbeddingManager
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = 'Crea embeddings de todos los productos en la base de datos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Tamaño de lote para procesar embeddings (default: 100)'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Fuerza recrear embeddings aunque ya existan'
        )

    def handle(self, *args, **options):
        start_time = datetime.now()

        self.stdout.write(
            self.style.SUCCESS('🚀 INICIANDO CREACIÓN DE EMBEDDINGS')
        )

        try:
            # Inicializar EmbeddingManager
            embedding_manager = EmbeddingManager()

            # Verificar si ya existen embeddings
            if not options['force']:
                try:
                    stats = embedding_manager.get_stats()
                    if stats.get('total_products', 0) > 0:
                        self.stdout.write(
                            self.style.WARNING(f'⚠️ Ya existen embeddings para {stats["total_products"]} productos')
                        )
                        self.stdout.write(
                            self.style.WARNING('Use --force para recrear o continúe con el entrenamiento')
                        )
                        return
                except:
                    pass  # No hay embeddings existentes, continuar

            # Crear embeddings
            batch_size = options['batch_size']
            self.stdout.write(f'🔄 Procesando productos en lotes de {batch_size}...')

            success = embedding_manager.create_embeddings_from_db(batch_size=batch_size)

            if success:
                # Mostrar estadísticas
                stats = embedding_manager.get_stats()
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds() / 60

                self.stdout.write(
                    self.style.SUCCESS('\n🎉 EMBEDDINGS CREADOS EXITOSAMENTE!')
                )
                self.stdout.write(f'⏱️  Tiempo total: {execution_time:.2f} minutos')
                self.stdout.write(f'📦 Total productos procesados: {stats["total_products"]}')
                self.stdout.write(f'🏷️  Categorías principales: {", ".join(list(stats["categories"].keys())[:5])}')
                self.stdout.write(
                    f'💰 Productos con descuento: {stats["products_with_discount"]} ({stats["discount_percentage"]})')

                # Probar búsqueda de ejemplo
                self.stdout.write(
                    self.style.SUCCESS('\n🔍 PRUEBA DE BÚSQUEDA:')
                )

                test_queries = ["celulares samsung", "computadores en oferta", "audífonos"]

                for query in test_queries:
                    results = embedding_manager.search_products(query, top_k=3)
                    self.stdout.write(f'\n📱 Búsqueda: "{query}"')
                    if results:
                        for i, product in enumerate(results, 1):
                            price_info = f"${product['price']:,.0f}"
                            if product['discount_percent'] != '0%':
                                price_info += f" ({product['discount_percent']} desc.)"
                            self.stdout.write(f'  {i}. {product["name"]} - {price_info}')
                    else:
                        self.stdout.write('  ❌ No se encontraron resultados')
            else:
                self.stdout.write(
                    self.style.ERROR('❌ Error creando embeddings')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error: {str(e)}')
            )
            raise

        # Mostrar siguiente paso
        self.stdout.write(
            self.style.SUCCESS('\n✅ SIGUIENTE PASO:')
        )
        self.stdout.write('Ejecute: python manage.py test_chatbot --interactive')
