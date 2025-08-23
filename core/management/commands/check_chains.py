from django.core.management.base import BaseCommand
from core.chatbot.bot import TechDiscountChatbot


class Command(BaseCommand):
    help = 'Verifica que todas las cadenas del chatbot estén funcionando'

    def handle(self, *args, **options):
        self.stdout.write("🔍 Verificando cadenas del chatbot...")

        try:
            bot = TechDiscountChatbot()
            self.stdout.write(
                self.style.SUCCESS('✅ Todas las cadenas inicializadas correctamente')
            )

            # Test rápido
            test_questions = [
                "Laptops con descuento",
                "Compara smartphones",
                "Busco tablets"
            ]

            for question in test_questions:
                response = bot.ask(question)
                self.stdout.write(f"❓ {question}")
                self.stdout.write(
                    f"✅ Respuesta: {len(response['answer'])} caracteres, {response['total_products']} productos")

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error: {str(e)}')
            )