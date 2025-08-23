from typing import Optional
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from core.Mongo.Schemas import ProductBase

class AlkostoScraper:
    def __init__(self):
        self.options = Options()
        self.options.add_argument("--headless")
        self.options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)

    def clean_price(self, price_str):
        """Convierte precios de texto a números"""
        if not price_str or price_str == "Sin descuento":
            return 0

        try:
            clean_str = price_str.replace('$', '').replace('.', '').replace(',', '').strip()
            return float(clean_str)
        except:
            return 0

    def extract_category_from_url(self, url):
        """Extrae categoría de la URL"""
        if not url:
            return "Sin categoría"

        try:
            # Mapeo de paths a categorías
            category_map = {
                'celulares/smartphones': 'Smartphones',
                'computadores-tablet/computadores-portatiles': 'Portátiles',
                'computadores-tablet/computadores-escritorio-all-in-one': 'Computadores de Escritorio',
                'computadores-tablet/tabletas-ipads': 'Tablets',
                'accesorios-electronica': 'Accesorios Electrónicos',
                'computadores-tablet/monitores': 'Monitores',
                'computadores-tablet/proyectores-videobeam': 'Proyectores',
                'tv/smart-tv': 'Televisores',
                'complementos-tv': 'Complementos TV',
                'accesorios-electronica/accesorios-tv-video': 'Accesorios TV',
                'videojuegos/consolas': 'Consolas',
                'videojuegos/accesorios-videojuegos': 'Accesorios Videojuegos',
                'casa-inteligente-domotica': 'Casa Inteligente'
            }

            for path, category in category_map.items():
                if path in url:
                    return category

            # Si no encuentra mapeo exacto, intenta inferir
            if 'celulares' in url: return 'Smartphones'
            if 'computadores' in url: return 'Computadores'
            if 'tablet' in url: return 'Tablets'
            if 'tv' in url: return 'Televisores'
            if 'videojuegos' in url: return 'Videojuegos'

            return "Electrónicos"
        except:
            return "Sin categoría"

    def get_content_selenium(self, url, clicks=3):
        """Obtiene contenido HTML de la URL con Selenium"""
        driver = webdriver.Chrome(options=self.options)

        try:
            print(f"🌐 Accediendo a: {url}")
            driver.get(url)

            # Esperar carga inicial
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "li.ais-InfiniteHits-item"))
                )
            except TimeoutException:
                print("Timeout: No se encontraron productos")
                return None, "No se encontraron productos"

            # Clicks en "Mostrar más"
            click_count = 0
            while click_count < clicks:
                try:
                    boton = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR,
                                                    "button.ais-InfiniteHits-loadMore.button-primary__outline.product__listing__load-more"))
                    )
                    driver.execute_script("arguments[0].click();", boton)
                    click_count += 1
                    print(f"📋 Click #{click_count} en 'Mostrar más'")
                    time.sleep(2)
                except Exception:
                    print(f"✅ Fin de los productos ({click_count} clicks realizados)")
                    break

            return driver.page_source, None

        except Exception as e:
            error_msg = f"❌ Error durante el scraping: {str(e)}"
            print(error_msg)
            return None, error_msg
        finally:
            driver.quit()

    def scrape_products(self, url, category=None):
        """Scrapea productos de una URL específica y devuelve objetos ProductBase"""
        html_content, error = self.get_content_selenium(url)

        if error:
            return [], error

        product_info_list = []
        soup = BeautifulSoup(html_content, 'html.parser')
        product_items = soup.find_all('li',
                                      class_='ais-InfiniteHits-item product__item js-product-item js-algolia-product-click')

        print(f"📊 Encontrados {len(product_items)} productos para scrapear")

        for item in product_items:
            try:
                product_data = self.extract_product_data(item, url, category)
                if product_data:
                    product_info_list.append(product_data)
            except Exception as e:
                print(f"⚠️ Error extrayendo producto: {e}")
                continue

        return product_info_list, None

    def extract_product_data(self, item, source_url, forced_category=None) -> Optional[ProductBase]:
        """Extrae datos de un producto individual y devuelve ProductBase"""
        # Extraer información básica
        name_tag = item.find('h3', class_=['product__item__top__title', 'js-algolia-product-click',
                                           'js-algolia-product-title'])
        if not name_tag:
            return None

        name = name_tag.get_text(strip=True)

        # URL del producto
        link_tag = item.find('a', class_='product__item__top__link')
        product_url = None
        if link_tag and link_tag.get('href'):
            href = link_tag['href']
            product_url = f"https://www.alkosto.com{href}" if href.startswith('/') else href

        # Marca
        brand_tag = item.find('div', class_='product__item__information__brand')
        brand = brand_tag.get_text(strip=True) if brand_tag else "Sin marca"

        # Descuento
        discount_percent_tag = item.find('span', class_='label-offer')
        discount_percent = discount_percent_tag.get_text(strip=True) if discount_percent_tag else "0%"

        # Rating
        stars_tag = item.find('span', class_='averageNumber')
        rating = stars_tag.get_text(strip=True) if stars_tag else "Sin calificación"

        # Precios
        old_price_tag = item.find('p', class_='product__price--discounts__old')
        discount_price_tag = item.find('span', class_='price')

        old_price_text = old_price_tag.get_text(strip=True) if old_price_tag else "Sin descuento"
        discount_price_text = discount_price_tag.get_text(strip=True) if discount_price_tag else "0"

        # Limpiar precios para valores numéricos
        original_price_num = self.clean_price(old_price_text)
        discount_price_num = self.clean_price(discount_price_text)

        # Imagen
        img_c_div = item.find('div', class_='product__item__information__image js-algolia-product-click')
        image_tag = img_c_div.find('img') if img_c_div else None
        image_url = ""
        if image_tag and image_tag.get('src'):
            src = image_tag['src']
            image_url = f"https://www.alkosto.com{src}" if src.startswith('/') else src

        # Especificaciones
        specifications = {}
        specs_container = item.find('ul', class_='product__item__information__key-features--list js-key-list')
        if specs_container:
            spec_items = specs_container.find_all('li', class_='item')
            for spec in spec_items:
                key_elem = spec.find('div', class_='item--key')
                value_elem = spec.find('div', class_='item--value')
                if key_elem and value_elem:
                    key = key_elem.get_text(strip=True)
                    value = value_elem.get_text(strip=True)
                    specifications[key] = value

        # Categoría
        category = forced_category if forced_category else self.extract_category_from_url(product_url or source_url)

        # Disponibilidad (inferir)
        availability = "Disponible"
        in_stock = "agotado" not in availability.lower()

        # Crear y retornar objeto ProductBase
        return ProductBase(
            name=name,
            brand=brand,
            category=category,
            product_url=product_url,
            source_url=source_url,
            discount_percent=discount_percent,
            rating=rating,
            original_price=old_price_text,
            original_price_num=original_price_num,
            discount_price=discount_price_text,
            discount_price_num=discount_price_num,
            image_url=image_url,
            specifications=specifications,
            availability=availability,
            in_stock=in_stock,
            source='alkosto'
        )