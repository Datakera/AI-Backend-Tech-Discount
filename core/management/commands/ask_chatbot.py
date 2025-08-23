from django.core.management.base import BaseCommand
from core.chatbot.bot import TechDiscountChatbot
import sys


class Command(BaseCommand):
    help = 'Hace una pregunta al chatbot de productos tecnológicos'

    def add_arguments(self, parser):
        parser.add_argument(
            'prompt',
            type=str,
            help='La pregunta para el chatbot (entre comillas si tiene espacios)'
        )
        parser.add_argument(
            '--verbose', '-d',  # ← CAMBIADO: -d en lugar de -v
            action='store_true',
            help='Muestra información detallada de los productos'
        )

    def handle(self, *args, **options):
        prompt = options['prompt']
        verbose = options['verbose']

        self.stdout.write(self.style.SUCCESS(f'🤖 Iniciando chatbot...'))

        try:
            # Inicializar chatbot
            bot = TechDiscountChatbot()

            self.stdout.write(self.style.SUCCESS(f'🧑‍💻 Pregunta: {prompt}'))
            self.stdout.write('-' * 80)

            # Hacer la pregunta
            response = bot.ask(prompt)

            # Mostrar respuesta
            self.stdout.write(self.style.SUCCESS('✅ Respuesta:'))
            self.stdout.write(response['answer'])

            # Mostrar productos si hay y es verbose
            if response['sources'] and verbose:
                self.stdout.write('\n' + '=' * 80)
                self.stdout.write(self.style.SUCCESS('📦 Productos encontrados:'))

                for i, product in enumerate(response['sources'], 1):
                    self.stdout.write(f'\n{i}. {self.style.BOLD(product["name"])}')
                    self.stdout.write(f'   🏷️  Marca: {product.get("brand", "N/A")}')
                    self.stdout.write(f'   💰 Precio: ${product.get("price", 0):,}')
                    self.stdout.write(f'   🎯 Descuento: {product.get("discount", "0%")}')
                    self.stdout.write(f'   🔗 URL: {product.get("url", "N/A")}')

            self.stdout.write('\n' + '=' * 80)
            self.stdout.write(self.style.SUCCESS(f'📊 Total productos relacionados: {response["total_products"]}'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))
            sys.exit(1)