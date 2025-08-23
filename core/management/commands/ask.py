# core/management/commands/ask.py
from django.core.management.base import BaseCommand
from core.chatbot.bot import TechDiscountChatbot


class Command(BaseCommand):
    help = 'Pregunta rápida al chatbot'

    def add_arguments(self, parser):
        parser.add_argument('pregunta', nargs='+', type=str, help='La pregunta para el chatbot')

    def handle(self, *args, **options):
        # Unir todas las palabras de la pregunta
        pregunta = ' '.join(options['pregunta'])

        bot = TechDiscountChatbot()
        respuesta = bot.ask(pregunta)

        print(f"🤖 {respuesta['answer']}")