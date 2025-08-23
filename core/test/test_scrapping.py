# core/tests/test_scrapping.py
import pymongo
from django.test import TestCase
from django.conf import settings
from core.mongo.Schemas import ProductBase
from core.scrapping.alkosto.Scrapping import AlkostoScraper
from core.mongo.MongoManager import MongoManager
import time


class AlkostoScraperTest(TestCase):
    """Pruebas exhaustivas para el scraper de Alkosto"""

    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.scraper = AlkostoScraper()
        self.mongo_manager = MongoManager(
            connection_string=settings.MONGODB_CONNECTION_STRING,
            db_name=settings.MONGODB_DB_NAME
        )

        # URLs de prueba para diferentes categorías
        self.test_urls = {
            'smartphones': 'https://www.alkosto.com/celulares/smartphones/c/BI_101_ALKOS',
            'laptops': 'https://www.alkosto.com/computadores-tablet/computadores-portatiles/c/BI_104_ALKOS',
            'tv': 'https://www.alkosto.com/tv/smart-tv/c/BI_120_ALKOS'
        }

    def tearDown(self):
        """Limpieza después de cada prueba"""
        self.mongo_manager.close_connection()

    def test_scraper_initialization(self):
        """Prueba que el scraper se inicialice correctamente"""
        self.assertIsNotNone(self.scraper)
        self.assertIsNotNone(self.scraper.options)
        print("✅ Scraper inicializado correctamente")

    def test_clean_price_method(self):
        """Prueba el método de limpieza de precios"""
        test_cases = [
            ('$1.000.000', 1000000),
            ('$2.500.000', 2500000),
            ('$999.999', 999999),
            ('Sin descuento', 0),
            ('', 0),
            (None, 0)
        ]

        for price_input, expected_output in test_cases:
            result = self.scraper.clean_price(price_input)
            self.assertEqual(result, expected_output)

        print("✅ Método clean_price funciona correctamente")

    def test_extract_category_from_url(self):
        """Prueba la extracción de categoría desde URLs"""
        test_cases = [
            ('https://www.alkosto.com/celulares/smartphones', 'Smartphones'),
            ('https://www.alkosto.com/computadores-tablet/computadores-portatiles', 'Portátiles'),
            ('https://www.alkosto.com/tv/smart-tv', 'Televisores'),
            ('https://www.alkosto.com/unknown-category', 'Electrónicos'),
            (None, 'Sin categoría'),
            ('', 'Sin categoría')
        ]

        for url, expected_category in test_cases:
            result = self.scraper.extract_category_from_url(url)
            self.assertEqual(result, expected_category)

        print("✅ Extracción de categoría funciona correctamente")

    def test_get_content_selenium_with_limited_clicks(self):
        """Prueba obtener contenido con número limitado de clicks (3)"""
        test_url = self.test_urls['smartphones']

        html_content, error = self.scraper.get_content_selenium(test_url, clicks=2)

        self.assertIsNone(error)
        self.assertIsNotNone(html_content)
        self.assertIn('ais-InfiniteHits-item', html_content)

        print("✅ Obtención de contenido con clicks limitados exitosa")

    def test_get_content_selenium_with_none_clicks(self):
        """Prueba obtener contenido hasta el final (clicks=None)"""
        test_url = self.test_urls['laptops']

        html_content, error = self.scraper.get_content_selenium(test_url, clicks=None)

        self.assertIsNone(error)
        self.assertIsNotNone(html_content)
        self.assertIn('ais-InfiniteHits-item', html_content)

        print("✅ Obtención de contenido hasta el final exitosa")

    def test_scrape_products_smoke_test(self):
        """Prueba básica de scraping de productos (smoke test)"""
        test_url = self.test_urls['tv']

        products, error = self.scraper.scrape_products(test_url, 'Televisores')

        self.assertIsNone(error)
        self.assertIsInstance(products, list)

        # Debería obtener al menos algunos productos
        self.assertGreater(len(products), 5)
        print(f"✅ Smoke test exitoso: {len(products)} productos encontrados")

    def test_scrape_products_with_pydantic_validation(self):
        """Prueba que los productos sean válidos según el schema Pydantic"""
        test_url = self.test_urls['smartphones']

        products, error = self.scraper.scrape_products(test_url, 'Smartphones')

        self.assertIsNone(error)

        for product in products[:3]:  # Probar solo los primeros 3
            # Verificar que es una instancia de ProductBase
            self.assertIsInstance(product, ProductBase)

            # Verificar campos obligatorios
            self.assertIsNotNone(product.name)
            self.assertIsNotNone(product.product_url)
            self.assertIsNotNone(product.source_url)
            self.assertEqual(product.source, 'alkosto')

            # Verificar que los precios numéricos son válidos
            self.assertGreaterEqual(product.original_price_num, 0)
            self.assertGreaterEqual(product.discount_price_num, 0)

        print("✅ Validación Pydantic exitosa para productos")

    def test_end_to_end_scraping_and_saving(self):
        """Prueba completa: scraping + guardado en MongoDB con visualización"""
        test_url = self.test_urls['laptops']
        category = 'Portátiles'

        # Scrapear productos
        products, error = self.scraper.scrape_products(test_url, category)
        self.assertIsNone(error)
        self.assertGreater(len(products), 0)

        print(f"📊 Productos scrapeados: {len(products)}")

        # Mostrar info del primer producto scrapeado
        if products:
            first_product = products[0]
            print("\n📦 PRIMER PRODUCTO SCRAPEADO:")
            print(f"   Nombre: {first_product.name}")
            print(f"   Marca: {first_product.brand}")
            print(f"   Precio original: {first_product.original_price}")
            print(f"   Precio con descuento: {first_product.discount_price}")
            print(f"   Descuento: {first_product.discount_percent}")
            print(f"   URL: {first_product.product_url}")

        # Guardar en MongoDB
        saved_count = self.mongo_manager.save_products(products, category)
        self.assertEqual(saved_count, len(products))

        print(f"💾 Productos guardados en MongoDB: {saved_count}")

        # Verificar que se guardaron correctamente
        saved_products = self.mongo_manager.get_products_by_category(category, 10)
        self.assertEqual(len(saved_products), min(10, len(products)))

        # MOSTRAR PRODUCTOS GUARDADOS EN CONSOLA
        print(f"\n🎯 PRODUCTOS GUARDADOS EN MONGODB ({len(saved_products)}):")
        print("═" * 80)

        for i, saved_product in enumerate(saved_products[:3]):  # Mostrar solo primeros 3
            print(f"\n📦 PRODUCTO #{i + 1} EN MONGODB:")
            print(f"   ID: {saved_product.id}")
            print(f"   Nombre: {saved_product.name}")
            print(f"   Marca: {saved_product.brand}")
            print(f"   Categoría: {saved_product.category}")
            print(f"   Precio original: {saved_product.original_price} (${saved_product.original_price_num:,.0f})")
            print(f"   Precio descuento: {saved_product.discount_price} (${saved_product.discount_price_num:,.0f})")
            print(f"   Descuento: {saved_product.discount_percent}")
            print(f"   Rating: {saved_product.rating}")
            print(f"   URL producto: {saved_product.product_url}")
            print(f"   URL fuente: {saved_product.source_url}")
            print(f"   Fuente: {saved_product.source}")
            print(f"   Fecha scraping: {saved_product.scraping_date}")
            print(f"   Disponibilidad: {saved_product.availability}")
            print(f"   En stock: {saved_product.in_stock}")

            # Mostrar algunas especificaciones
            if saved_product.specifications:
                print(f"   Especificaciones:")
                for key, value in list(saved_product.specifications.items())[:3]:  # Primeras 3 specs
                    print(f"     - {key}: {value}")
                if len(saved_product.specifications) > 3:
                    print(f"     - ... y {len(saved_product.specifications) - 3} más")

            print("─" * 60)

        # Verificaciones adicionales
        for saved_product in saved_products:
            self.assertEqual(saved_product.category, category)
            self.assertEqual(saved_product.source, 'alkosto')
            self.assertIsNotNone(saved_product.id)

            # Verificar que los datos numéricos son válidos
            self.assertGreaterEqual(saved_product.original_price_num, 0)
            self.assertGreaterEqual(saved_product.discount_price_num, 0)

            self.assertLessEqual(saved_product.discount_price_num, saved_product.original_price_num)

        print(f"\n✅ End-to-end test exitoso: {saved_count} productos guardados")

        # ESTADÍSTICAS FINALES
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   Total productos scrapeados: {len(products)}")
        print(f"   Total productos guardados: {saved_count}")
        print(f"   Productos mostrados: {min(3, len(saved_products))}")

        # Mostrar rango de precios
        if saved_products:
            prices = [p.discount_price_num for p in saved_products if p.discount_price_num > 0]
            if prices:
                print(f"   Precio mínimo: ${min(prices):,.0f}")
                print(f"   Precio máximo: ${max(prices):,.0f}")
                print(f"   Precio promedio: ${sum(prices) / len(prices):,.0f}")

        # Limpiar los productos de prueba (opcional)
        self.cleanup_test_products(category)

    def cleanup_test_products(self, category):
        """Limpia los productos de prueba de la base de datos"""
        try:
            result = self.mongo_manager.products_collection.delete_many({
                'category': category,
                'scraping_date': {'$gte': '2024'}  # Solo productos recientes del test
            })
            print(f"\n🧹 LIMPIEZA: {result.deleted_count} productos de prueba eliminados")
        except Exception as e:
            print(f"⚠️  Error en limpieza: {e}")

    def test_scraping_performance(self):
        """Prueba de performance del scraping"""
        test_url = self.test_urls['smartphones']

        start_time = time.time()
        products, error = self.scraper.scrape_products(test_url, 'Smartphones')
        end_time = time.time()

        self.assertIsNone(error)
        execution_time = end_time - start_time

        print(f"⏱️ Tiempo de scraping: {execution_time:.2f} segundos")
        print(f"📊 Productos por segundo: {len(products) / execution_time:.2f}")

        # El scraping debería tomar menos de 60 segundos
        self.assertLess(execution_time, 60)
        print("✅ Test de performance exitoso")

    def test_error_handling_invalid_url(self):
        """Prueba manejo de errores con URL inválida"""
        invalid_url = "https://www.alkosto.com/url-que-no-existe"

        html_content, error = self.scraper.get_content_selenium(invalid_url)

        self.assertIsNotNone(error)
        self.assertIsNone(html_content)
        print("✅ Manejo de errores con URL inválida funciona")

    def test_scraping_multiple_categories(self):
        """Prueba scraping de múltiples categorías"""
        categories_to_test = ['smartphones', 'laptops', 'tv']

        all_products = []
        for category_name in categories_to_test:
            test_url = self.test_urls[category_name]

            products, error = self.scraper.scrape_products(test_url, category_name)
            self.assertIsNone(error)
            all_products.extend(products)

            print(f"✅ {category_name}: {len(products)} productos")
            time.sleep(1)  # Pausa entre categorías

        self.assertGreater(len(all_products), 0)
        print(f"✅ Scraping múltiple exitoso: {len(all_products)} productos totales")

    def test_product_data_quality(self):
        """Prueba la calidad de los datos scrapeados"""
        test_url = self.test_urls['tv']

        products, error = self.scraper.scrape_products(test_url, 'Televisores')
        self.assertIsNone(error)

        for product in products[:5]:  # Revisar los primeros 5 productos
            # Verificar que los nombres no estén vacíos
            self.assertTrue(len(product.name.strip()) > 0)

            # Verificar que las URLs son válidas
            self.assertTrue(product.product_url.startswith('http'))
            self.assertTrue(product.source_url.startswith('http'))

            # Verificar que los precios son consistentes
            if product.original_price_num > 0 and product.discount_price_num > 0:
                self.assertLessEqual(product.discount_price_num, product.original_price_num)

            # Verificar que el discount_percent es un string con %
            self.assertIn('%', product.discount_percent)

        print("✅ Calidad de datos verificada exitosamente")


# Tests para el MongoManager
class MongoManagerTest(TestCase):
    """Pruebas específicas para el MongoManager"""

    def setUp(self):
        self.mongo_manager = MongoManager(
            connection_string=settings.MONGODB_CONNECTION_STRING,
            db_name=settings.MONGODB_DB_NAME
        )

    def tearDown(self):
        self.mongo_manager.close_connection()

    def test_save_and_retrieve_product(self):
        """Prueba guardar y recuperar un producto de prueba con Pydantic"""
        try:
            from core.mongo.Schemas import ProductBase

            # Crear producto con Pydantic
            test_product = ProductBase(
                name='Producto de Prueba Pydantic',
                brand='Marca Test',
                category='Test Category',
                product_url='https://www.alkosto.com/test-product-pydantic',
                source_url='https://www.alkosto.com/test-category',
                discount_percent='25%',
                rating='4.8',
                original_price='$1.200.000',
                original_price_num=1200000,
                discount_price='$900.000',
                discount_price_num=900000,
                image_url='https://example.com/image.jpg',
                specifications={'RAM': '16GB', 'Storage': '512GB'},
                availability='Disponible',
                in_stock=True,
                source='alkosto'
            )

            print(f"📝 Producto creado: {test_product.name}")

            # Guardar producto - DEBUG
            print("💾 Intentando guardar producto...")
            saved_count = self.mongo_manager.save_products([test_product], 'Test Category')
            print(f"📊 Resultado de save_products: {saved_count}")

            # Verificar que se guardó
            self.assertEqual(saved_count, 1, f"Expected 1, got {saved_count}")

            # DEBUG: Verificar si está en la base de datos directamente
            direct_check = self.mongo_manager.products_collection.find_one({
                'product_url': 'https://www.alkosto.com/test-product-pydantic'
            })
            print(f"🔍 Búsqueda directa en MongoDB: {direct_check is not None}")

            if direct_check:
                print(f"📄 Documento encontrado: {direct_check['name']}")

            # Recuperar producto a través del manager
            retrieved_product = self.mongo_manager.get_product_by_url(
                'https://www.alkosto.com/test-product-pydantic'
            )
            print(f"🔍 Producto recuperado: {retrieved_product is not None}")

            # Verificar que los datos coinciden
            self.assertIsNotNone(retrieved_product, "El producto no se recuperó de MongoDB")
            self.assertEqual(retrieved_product.name, 'Producto de Prueba Pydantic')
            self.assertEqual(retrieved_product.brand, 'Marca Test')
            self.assertEqual(retrieved_product.discount_percent, '25%')

            print("✅ Guardado y recuperación con Pydantic exitoso")

            # Limpiar
            delete_result = self.mongo_manager.products_collection.delete_one({
                'product_url': 'https://www.alkosto.com/test-product-pydantic'
            })
            print(f"🧹 Producto eliminado: {delete_result.deleted_count}")

        except pymongo.errors.ServerSelectionTimeoutError:
            self.skipTest("MongoDB no está disponible para testing")
        except Exception as e:
            print(f"❌ Error durante el test: {e}")
            raise