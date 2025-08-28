from django.core.management.base import BaseCommand
from core.chatbot.TechChatbot import TechChatbot  # Importar nueva clase
import os
import logging

logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = 'Prueba el chatbot ligero de productos tecnológicos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Modo interactivo de conversación'
        )
        parser.add_argument(
            '--base-model-only',
            action='store_true',
            help='Usar solo el modelo base (sin LoRA fine-tuning)'
        )
        parser.add_argument(
            '--model',
            type=str,
            choices=['extra_small', 'small', 'medium'],
            default='small',
            help='Tamaño del modelo a probar'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            default='models/lightweight_chatbot',
            help='Ruta del modelo LoRA entrenado'
        )
        parser.add_argument(
            '--save-conversation',
            action='store_true',
            help='Guardar la conversación en un archivo'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🤖 PROBANDO CHATBOT LIGERO')
        )

        # Mapeo de modelos
        model_map = {
            'extra_small': 'distilgpt2',
            'small': 'microsoft/DialoGPT-small',
            'medium': 'microsoft/DialoGPT-medium'
        }

        model_name = model_map[options['model']]

        try:
            # Verificar si existe modelo entrenado
            model_exists = os.path.exists(options['model_path'])
            if not model_exists and not options['base_model_only']:
                self.stdout.write(
                    self.style.WARNING('No se encontró modelo LoRA entrenado')
                )
                self.stdout.write('Usando modelo base. Para entrenar ejecute: python manage.py train_chatbot')
                options['base_model_only'] = True

            # Inicializar chatbot ligero
            self.stdout.write(f'Cargando chatbot con {model_name}...')
            chatbot = TechChatbot(
                base_model_name=model_name,
                lora_path=options['model_path']
            )
            chatbot.load_model(load_base_only=options['base_model_only'])

            self.stdout.write('Chatbot cargado correctamente')

            # Mostrar información
            model_info = "Modelo base" if options['base_model_only'] else "Modelo con LoRA fine-tuning"
            self.stdout.write(f'Usando: {model_info}')
            self.stdout.write(f'Tipo: {chatbot.model_type}')

            # Obtener estadísticas de embeddings
            try:
                stats = chatbot.embedding_manager.get_stats()
                self.stdout.write(f'Productos en índice: {stats.get("total_products", "N/A")}')
            except:
                self.stdout.write('No se pudieron cargar estadísticas de productos')

            # Modo de operación
            if options['interactive']:
                self._interactive_mode(chatbot, options['save_conversation'])
            else:
                self._run_lightweight_tests(chatbot)

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error: {str(e)}')
            )

            # Sugerencias específicas para modelos ligeros
            if 'memory' in str(e).lower() or 'cuda' in str(e).lower():
                self.stdout.write('\nSUGERENCIAS:')
                self.stdout.write('• Usa --model extra_small')
                self.stdout.write('• El modelo DialoGPT-small debería funcionar en la mayoría de PCs')
                self.stdout.write('• Verifica que tengas al menos 2GB de RAM disponible')

    def _interactive_mode(self, chatbot, save_conversation):
        """Modo interactivo optimizado para modelos ligeros"""
        self.stdout.write(
            self.style.SUCCESS('\n💬 MODO INTERACTIVO - CHATBOT LIGERO')
        )
        self.stdout.write('Comandos especiales: "salir", "limpiar", "stats"')
        self.stdout.write('=' * 50)

        try:
            while True:
                user_input = input('\n👤 Tú: ').strip()

                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break
                elif user_input.lower() in ['limpiar', 'clear', 'reset']:
                    chatbot.clear_conversation()
                    self.stdout.write('Conversación reiniciada')
                    continue
                elif user_input.lower() in ['stats', 'estadisticas']:
                    stats = chatbot.get_conversation_stats()
                    self.stdout.write(f'Interacciones: {stats["total_interactions"]}')
                    self.stdout.write(f'Búsquedas exitosas: {stats["success_rate"]}')
                    continue

                if not user_input:
                    continue

                # Obtener respuesta del chatbot
                self.stdout.write('🤖 Procesando...', ending='')
                response = chatbot.chat(user_input)
                self.stdout.write('\r🤖 Bot: ' + response)

        except KeyboardInterrupt:
            self.stdout.write('\n\nConversación terminada por el usuario')

        # Mostrar estadísticas finales
        stats = chatbot.get_conversation_stats()
        if stats['total_interactions'] > 0:
            self.stdout.write(
                self.style.SUCCESS('\nESTADÍSTICAS DE LA CONVERSACIÓN:')
            )
            self.stdout.write(f'Total interacciones: {stats["total_interactions"]}')
            self.stdout.write(f'Búsquedas exitosas: {stats["success_rate"]}')

            # Guardar conversación si se solicita
            if save_conversation:
                try:
                    filepath = chatbot.save_conversation()
                    self.stdout.write(f'Conversación guardada en: {filepath}')
                except Exception as e:
                    self.stdout.write(f'Error guardando conversación: {str(e)}')

    def _run_lightweight_tests(self, chatbot):
        """Ejecuta pruebas optimizadas para modelos ligeros"""
        test_cases = [
            # Pruebas básicas
            {
                'category': 'SALUDOS BÁSICOS',
                'queries': [
                    'Hola',
                    'Qué productos tienes',
                    'Ayuda'
                ]
            },
            # Búsquedas simples
            {
                'category': 'BÚSQUEDAS SIMPLES',
                'queries': [
                    'celulares',
                    'computadores',
                    'ofertas'
                ]
            },
            # Búsquedas por marca
            {
                'category': 'BÚSQUEDAS POR MARCA',
                'queries': [
                    'Samsung',
                    'Apple',
                    'productos Xiaomi'
                ]
            },
            # Ofertas específicas
            {
                'category': 'OFERTAS',
                'queries': [
                    'descuentos',
                    'celulares baratos',
                    'ofertas Samsung'
                ]
            }
        ]

        self.stdout.write(
            self.style.SUCCESS('\nEJECUTANDO PRUEBAS LIGERAS')
        )

        total_tests = sum(len(test_case['queries']) for test_case in test_cases)
        current_test = 0

        for test_case in test_cases:
            self.stdout.write(
                self.style.WARNING(f'\n--- {test_case["category"]} ---')
            )

            for query in test_case['queries']:
                current_test += 1
                self.stdout.write(f'\n[{current_test}/{total_tests}] 👤 Usuario: {query}')

                try:
                    response = chatbot.chat(query)
                    # Mostrar respuesta limitada para legibilidad
                    display_response = response[:100] + "..." if len(response) > 100 else response
                    self.stdout.write(f'🤖 Bot: {display_response}')

                    # Pausa breve
                    import time
                    time.sleep(0.5)

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Error en consulta: {str(e)}')
                    )

        # Estadísticas finales
        stats = chatbot.get_conversation_stats()
        self.stdout.write(
            self.style.SUCCESS('\nESTADÍSTICAS DE PRUEBA:')
        )
        self.stdout.write(f'Pruebas completadas: {total_tests}')
        self.stdout.write(f'Total interacciones: {stats["total_interactions"]}')
        self.stdout.write(f'Búsquedas exitosas: {stats["success_rate"]}')

        # Evaluación simplificada
        self._evaluate_lightweight_performance(chatbot, stats)

    def _evaluate_lightweight_performance(self, chatbot, stats):
        """Evaluación básica para modelos ligeros"""
        self.stdout.write(
            self.style.SUCCESS('\nEVALUACIÓN DE RENDIMIENTO:')
        )

        # Prueba de embeddings
        try:
            search_results = chatbot.embedding_manager.search_products("celular samsung", top_k=3)
            embeddings_ok = "✅" if search_results else "❌"
            self.stdout.write(f'Búsqueda embeddings: {embeddings_ok} ({len(search_results)} resultados)')
        except:
            self.stdout.write('Búsqueda embeddings: ❌ Error')

        # Calidad de respuestas
        success_rate = float(stats['success_rate'].replace('%', ''))
        if success_rate >= 70:
            quality = "✅ Excelente"
        elif success_rate >= 50:
            quality = "⚠️ Aceptable"
        else:
            quality = "❌ Necesita mejoras"

        self.stdout.write(f'Calidad de respuestas: {quality}')

        # Recomendaciones
        self.stdout.write(
            self.style.SUCCESS('\nRECOMENCIACIONES:')
        )

        if success_rate < 50:
            self.stdout.write('• Considera reentrenar con más epochs')
            self.stdout.write('• Verifica que los embeddings estén correctamente creados')
        elif success_rate < 80:
            self.stdout.write('• El rendimiento es bueno para un modelo ligero')
            self.stdout.write('• Puedes probar con --model medium para mejor calidad')
        else:
            self.stdout.write('• ¡Excelente rendimiento para un modelo ligero!')
            self.stdout.write('• El chatbot está listo para uso en producción')

        self.stdout.write('• Para más pruebas usa: --interactive')