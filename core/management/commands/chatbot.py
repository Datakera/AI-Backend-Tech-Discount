from django.core.management.base import BaseCommand
from core.chatbot.bot import TechDiscountChatbot
import sys


class Command(BaseCommand):
    help = 'Chatbot interactivo para productos tecnológicos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--question', '-q',
            type=str,
            help='Pregunta específica para el chatbot'
        )
        parser.add_argument(
            '--file', '-f',
            type=str,
            help='Archivo con preguntas (una por línea)'
        )
        parser.add_argument(
            '--interactive', '-i',
            action='store_true',
            help='Modo interactivo'
        )

    def handle(self, *args, **options):
        if options['question']:
            self.ask_question(options['question'])
        elif options['file']:
            self.ask_from_file(options['file'])
        elif options['interactive']:
            self.interactive_mode()
        else:
            self.stdout.write(self.style.WARNING('💡 Usa --help para ver opciones disponibles'))

    def ask_question(self, question):
        """Hacer una pregunta específica"""
        bot = TechDiscountChatbot()
        response = bot.ask(question)

        self.stdout.write(self.style.SUCCESS(f'\n❓ {question}'))
        self.stdout.write('-' * 60)
        self.stdout.write(f'🤖 {response["answer"]}')
        self.stdout.write(f'📦 Productos encontrados: {response["total_products"]}')

    def ask_from_file(self, filename):
        """Hacer preguntas desde un archivo"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

            bot = TechDiscountChatbot()

            for question in questions:
                response = bot.ask(question)
                self.stdout.write(f'\n❓ {question}')
                self.stdout.write(f'🤖 {response["answer"][:200]}...')
                self.stdout.write(f'📦 {response["total_products"]} productos')
                self.stdout.write('─' * 40)

        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'❌ Archivo no encontrado: {filename}'))

    def interactive_mode(self):
        """Modo interactivo"""
        self.stdout.write(self.style.SUCCESS('🎮 Modo interactivo - Chatbot de Productos'))
        self.stdout.write('💡 Escribe "salir" para terminar')
        self.stdout.write('=' * 50)

        bot = TechDiscountChatbot()

        while True:
            try:
                question = input('\n🧑‍💻 Tú: ').strip()

                if question.lower() in ['salir', 'exit', 'quit', 'q']:
                    break

                if not question:
                    continue

                response = bot.ask(question)

                print(f'🤖 Chatbot: {response["answer"]}')
                print(f'   📦 {response["total_products"]} productos relacionados')

            except KeyboardInterrupt:
                self.stdout.write(self.style.WARNING('\n👋 ¡Hasta luego!'))
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'❌ Error: {str(e)}'))