from django.core.management.base import BaseCommand
from core.chatbot.ChatbotTrainer import ChatbotTrainer
from datetime import datetime
import logging
import torch

logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = 'Entrena el chatbot para productos tecnológicos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=3,
            help='Número de epochs (default: 3)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=4,
            help='Batch size (default: 4)'
        )
        parser.add_argument(
            '--max-samples',
            type=int,
            default=5000,
            help='Máximo número de muestras del dataset (default: 5000)'
        )
        parser.add_argument(
            '--cpu-only',
            action='store_true',
            help='Forzar uso de CPU únicamente'
        )

    def handle(self, *args, **options):
        start_time = datetime.now()

        self.stdout.write(
            self.style.SUCCESS('🚀 ENTRENAMIENTO DE CHATBOT TECNOLÓGICO')
        )

        # Mostrar configuración del sistema
        if torch.cuda.is_available() and not options['cpu_only']:
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.stdout.write(f'GPU disponible: {gpu_name} ({gpu_memory:.1f}GB)')
        else:
            self.stdout.write('Usando CPU para entrenamiento')

        # Inicializar trainer
        self.stdout.write('🔄 Inicializando trainer...')
        trainer = ChatbotTrainer(model_name="microsoft/DialoGPT-medium")

        # Crear dataset
        self.stdout.write('📊 Creando dataset de entrenamiento...')
        dataset = trainer.create_training_dataset(max_samples=options['max_samples'])
        self.stdout.write(f'✅ Dataset creado: {len(dataset)} ejemplos')

        # Cargar modelo
        self.stdout.write('🔄 Cargando modelo...')
        trainer.load_model()

        # Mostrar configuración
        self.stdout.write(self.style.SUCCESS('\n📋 CONFIGURACIÓN:'))
        self.stdout.write(f'🤖 Modelo: Microsoft DialoGPT-medium')
        self.stdout.write(f'📊 Ejemplos de entrenamiento: {len(dataset)}')
        self.stdout.write(f'🔄 Epochs: {options["epochs"]}')
        self.stdout.write(f'📦 Batch size: {options["batch_size"]}')

        # Estimar tiempo
        estimated_minutes = self._estimate_training_time(len(dataset), options['epochs'])
        self.stdout.write(f'⏱️ Tiempo estimado: {estimated_minutes:.1f} minutos')

        # Entrenar
        self.stdout.write(self.style.SUCCESS('\n🚀 INICIANDO ENTRENAMIENTO...'))
        trainer.train(
            dataset=dataset,
            epochs=options['epochs'],
            batch_size=options['batch_size']
        )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60

        self.stdout.write(self.style.SUCCESS(f'\n🎉 ENTRENAMIENTO COMPLETADO!'))
        self.stdout.write(f'⏱️ Tiempo real: {training_time:.1f} minutos')
        self.stdout.write(f'💾 Modelo guardado en: {trainer.output_dir}')

        # Mostrar recomendaciones
        self.stdout.write(self.style.SUCCESS('\n✅ RECOMENDACIONES:'))
        self.stdout.write('• Ejecuta: python manage.py test_chatbot --interactive')
        self.stdout.write('• Para mejor rendimiento, usa --epochs 4-5 con más datos')
        self.stdout.write('• El modelo está optimizado para 1,200-10,000 productos')

    def _estimate_training_time(self, samples, epochs):
        """Estima tiempo de entrenamiento basado en samples y epochs"""
        # Estimación conservadora: 0.2 segundos por sample por epoch en CPU
        total_seconds = samples * epochs * 0.2
        return total_seconds / 60  # Convertir a minutos