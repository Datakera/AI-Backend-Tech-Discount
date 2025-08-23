# core/management/commands/train_chatbot.py
from django.core.management.base import BaseCommand
from core.chatbot.training.train import train_chatbot


class Command(BaseCommand):
    help = 'Entrena el chatbot con los productos de la base de datos'

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🚀 Entrenando chatbot...')
        )

        success = train_chatbot()

        if success:
            self.stdout.write(
                self.style.SUCCESS('✅ Chatbot entrenado exitosamente!')
            )
        else:
            self.stdout.write(
                self.style.ERROR('❌ Error entrenando el chatbot')
            )