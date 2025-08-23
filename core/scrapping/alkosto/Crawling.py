import time

from core.mongo.MongoManager import MongoManager
from core.scrapping.alkosto.Scrapping import AlkostoScraper


class AlkostoCrawler:
    def __init__(self, clicks = None):
        self.scraper = AlkostoScraper()
        self.mongo_manager = MongoManager()
        self.clicks = clicks
        self.category_urls = {
            'smartphones': 'https://www.alkosto.com/celulares/smartphones/c/BI_101_ALKOS',
            'portatiles': 'https://www.alkosto.com/computadores-tablet/computadores-portatiles/c/BI_104_ALKOS',
            'computadores_escritorio': 'https://www.alkosto.com/computadores-tablet/computadores-escritorio-all-in-one/c/BI_105_ALKOS',
            'tablets': 'https://www.alkosto.com/computadores-tablet/tabletas-ipads/c/BI_107_ALKOS',
            'accesorios_electronicos': 'https://www.alkosto.com/accesorios-electronica/c/BI_AELE_ALKOS',
            'monitores': 'https://www.alkosto.com/computadores-tablet/monitores/c/BI_110_ALKOS',
            'proyectores': 'https://www.alkosto.com/computadores-tablet/proyectores-videobeam/c/BI_121_ALKOS',
            'televisores': 'https://www.alkosto.com/tv/smart-tv/c/BI_120_ALKOS',
            'complementos_tv': 'https://www.alkosto.com/complementos-tv/c/complementos-tv',
            'accesorios_tv': 'https://www.alkosto.com/accesorios-electronica/accesorios-tv-video/c/BI_123_ALKOS',
            'consolas': 'https://www.alkosto.com/videojuegos/consolas/c/BI_131_ALKOS',
            'accesorios_videojuegos': 'https://www.alkosto.com/videojuegos/accesorios-videojuegos/c/BI_133_ALKOS',
            'casa_inteligente': 'https://www.alkosto.com/casa-inteligente-domotica/c/BI_CAIN_ALKOS'
        }

    def crawl_category(self, category_name, url):
        """Crawlea una categoría específica"""
        print(f"\n🚀 Iniciando crawling de: {category_name}")
        print(f"📁 URL: {url}")
        print(f"🔢 Modo: {'TODOS los productos' if self.clicks is None else f'{self.clicks} clicks'}")

        # Pasar el parámetro clicks al scraper
        products, error = self.scraper.scrape_products(url, category_name, clicks=self.clicks)

        if error:
            print(f"❌ Error en {category_name}: {error}")
            return []

        print(f"✅ {len(products)} productos con descuento encontrados en {category_name}")

        # Guardar en MongoDB
        if products:
            saved_count = self.mongo_manager.save_products(products, category_name)
            print(f"💾 {saved_count} productos guardados en MongoDB")

        return products

    def crawl_all_categories(self):
        """Crawlea todas las categorías"""
        all_products = []

        for category_name, url in self.category_urls.items():
            products = self.crawl_category(category_name, url)
            all_products.extend(products)
            time.sleep(2)  # Pausa entre categorías

        print(f"\n🎉 Crawling completado! Total: {len(all_products)} productos con descuento")
        return all_products

    def crawl_specific_categories(self, categories):
        """Crawlea categorías específicas"""
        results = {}
        for category in categories:
            if category in self.category_urls:
                products = self.crawl_category(category, self.category_urls[category])
                results[category] = products
            else:
                print(f"⚠️ Categoría no encontrada: {category}")

        return results