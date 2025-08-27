from django.core.management.base import BaseCommand
from core.chatbot.ChatbotTrainer import ChatbotTrainer
from datetime import datetime
import logging
import torch
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = 'Entrena el chatbot con LoRA fine-tuning'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=3,
            help='Número de epochs de entrenamiento (default: 3)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=4,
            help='Tamaño de batch para entrenamiento (default: 4)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=2e-4,
            help='Learning rate (default: 2e-4)'
        )
        parser.add_argument(
            '--model',
            type=str,
            default='mistralai/Mistral-7B-Instruct-v0.2',
            help='Modelo base a usar (default: Mistral-7B-Instruct-v0.2)'
        )
        parser.add_argument(
            '--skip-training',
            action='store_true',
            help='Omite el entrenamiento y solo prepara el dataset'
        )
        parser.add_argument(
            '--test-model',
            action='store_true',
            help='Prueba el modelo después del entrenamiento'
        )

    def handle(self, *args, **options):
        start_time = datetime.now()

        self.stdout.write(
            self.style.SUCCESS('🤖 INICIANDO ENTRENAMIENTO DEL CHATBOT')
        )

        # Verificar GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.stdout.write(f'🔥 GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)')
        else:
            self.stdout.write(
                self.style.WARNING('⚠️ No se detectó GPU. El entrenamiento será muy lento.')
            )

        try:
            # Inicializar trainer
            trainer = ChatbotTrainer(model_name=options['model'])

            # Crear dataset
            self.stdout.write('📊 Creando dataset de entrenamiento...')
            dataset = trainer.create_training_dataset()

            self.stdout.write(f'✅ Dataset creado con {len(dataset)} ejemplos')

            # Cargar modelo
            self.stdout.write('🔄 Cargando modelo...')
            trainer.load_model()

            if not options['skip_training']:
                # Configuración de entrenamiento
                config = {
                    'model_name': options['model'],
                    'epochs': options['epochs'],
                    'batch_size': options['batch_size'],
                    'learning_rate': options['learning_rate'],
                    'dataset_size': len(dataset),
                    'training_date': datetime.now().isoformat()
                }

                # Mostrar configuración
                self.stdout.write(
                    self.style.SUCCESS('\n📋 CONFIGURACIÓN DE ENTRENAMIENTO:')
                )
                self.stdout.write(f'🏷️  Modelo: {config["model_name"]}')
                self.stdout.write(f'🔄 Epochs: {config["epochs"]}')
                self.stdout.write(f'📦 Batch size: {config["batch_size"]}')
                self.stdout.write(f'📈 Learning rate: {config["learning_rate"]}')
                self.stdout.write(f'📊 Ejemplos de entrenamiento: {config["dataset_size"]}')

                # Guardar configuración
                trainer.save_training_config(config)

                # Entrenar
                self.stdout.write(
                    self.style.SUCCESS('\n🚀 INICIANDO ENTRENAMIENTO...')
                )
                self.stdout.write('⚠️ Esto puede tomar entre 30-60 minutos dependiendo de tu GPU')

                trainer.train(
                    dataset=dataset,
                    epochs=options['epochs'],
                    batch_size=options['batch_size']
                )

                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds() / 60

                self.stdout.write(
                    self.style.SUCCESS(f'\n🎉 ENTRENAMIENTO COMPLETADO!')
                )
                self.stdout.write(f'⏱️  Tiempo total: {training_time:.2f} minutos')
                self.stdout.write(f'💾 Modelo guardado en: {trainer.output_dir}')

                # Probar modelo si se solicita
                if options['test_model']:
                    self._test_trained_model(trainer.output_dir)
            else:
                self.stdout.write(
                    self.style.SUCCESS('✅ Dataset preparado. Entrenamiento omitido.')
                )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error durante entrenamiento: {str(e)}')
            )
            raise

        # Mostrar siguiente paso
        self.stdout.write(
            self.style.SUCCESS('\n✅ SIGUIENTE PASO:')
        )
        self.stdout.write('Ejecute: python manage.py test_chatbot')

    def _test_trained_model(self, model_path: str):
        """Prueba rápida del modelo entrenado"""
        try:
            self.stdout.write(
                self.style.SUCCESS('\n🧪 PROBANDO MODELO ENTRENADO...')
            )

            from core.chatbot.TechChatbot import TechChatbot

            # Cargar chatbot
            chatbot = TechChatbot(lora_path=model_path)
            chatbot.load_model()

            # Pruebas básicas
            test_queries = [
                "Hola, ¿qué productos tienes?",
                "Busco un celular Samsung",
                "¿Hay computadores en oferta?"
            ]

            for query in test_queries:
                self.stdout.write(f'\n👤 Usuario: {query}')
                response = chatbot.chat(query)
                self.stdout.write(f'🤖 Bot: {response}')

        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f'⚠️ No se pudo probar el modelo: {str(e)}')
            )
