from django.core.management.base import BaseCommand
from core.scrapping.alkosto.Crawling import AlkostoCrawler
from datetime import datetime


class Command(BaseCommand):
    help = 'Ejecuta el scraping completo de Alkosto'

    def add_arguments(self, parser):
        parser.add_argument(
            '--categories',
            nargs='+',
            help='Categorías específicas a scrapear (opcional)'
        )
        parser.add_argument(
            '--clicks',
            type=int,
            default=None,  # ← Por defecto None para todos los productos
            help='Número de clicks en "Mostrar más" (None para todos)'
        )
        parser.add_argument(
            '--limit-categories',
            type=int,
            default=None,
            help='Límite de categorías a scrapear'
        )

    def handle(self, *args, **options):
        start_time = datetime.now()

        # Crear crawler con el parámetro clicks
        crawler = AlkostoCrawler(clicks=options['clicks'])
        categories = options['categories']
        limit_categories = options['limit_categories']

        self.stdout.write(
            self.style.SUCCESS('🚀 INICIANDO SCRAPING COMPLETO DE ALKOSTO')
        )

        if options['clicks'] is None:
            self.stdout.write("🔢 MODO: TODOS los productos (clicks=None)")
        else:
            self.stdout.write(f"🔢 MODO: {options['clicks']} clicks por categoría")

        if categories:
            # Scrapear categorías específicas
            results = crawler.crawl_specific_categories(categories)
            total_products = sum(len(products) for products in results.values())
        else:
            # Scrapear todas las categorías o limitar
            if limit_categories:
                # Limitar categorías
                limited_urls = dict(list(crawler.category_urls.items())[:limit_categories])
                total_products = 0
                for category_name, url in limited_urls.items():
                    products = crawler.crawl_category(category_name, url)
                    total_products += len(products)
            else:
                # Todas las categorías
                all_products = crawler.crawl_all_categories()
                total_products = len(all_products)

        # Estadísticas
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds() / 60

        self.stdout.write(
            self.style.SUCCESS(f'\n🎉 SCRAPING COMPLETADO!')
        )
        self.stdout.write(f'⏱️  Tiempo total: {execution_time:.2f} minutos')
        self.stdout.write(f'📦 Total productos con descuento: {total_products}')

        # Verificar base de datos
        from core.mongo.MongoManager import MongoManager
        mongo = MongoManager()
        self.stdout.write(f'📊 Total en MongoDB: {mongo.get_product_count()} productos')
        self.stdout.write(f'🏷️  Categorías: {", ".join(mongo.get_categories())}')