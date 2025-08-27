from django.core.management.base import BaseCommand
from core.chatbot.TechChatbot import TechChatbot
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = 'Prueba el chatbot de productos tecnológicos'

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
            '--test-queries',
            nargs='+',
            help='Lista de consultas específicas para probar'
        )
        parser.add_argument(
            '--model-path',
            type=str,
            default='models/chatbot_lora',
            help='Ruta del modelo LoRA entrenado'
        )
        parser.add_argument(
            '--save-conversation',
            action='store_true',
            help='Guardar la conversación en un archivo'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('🤖 INICIANDO PRUEBA DEL CHATBOT')
        )

        try:
            # Verificar si existe modelo entrenado
            model_exists = os.path.exists(options['model_path'])
            if not model_exists and not options['base_model_only']:
                self.stdout.write(
                    self.style.WARNING('⚠️ No se encontró modelo LoRA entrenado')
                )
                self.stdout.write('Usando modelo base. Para entrenar ejecute: python manage.py train_chatbot')
                options['base_model_only'] = True

            # Inicializar chatbot
            self.stdout.write('🔄 Cargando chatbot...')
            chatbot = TechChatbot(lora_path=options['model_path'])
            chatbot.load_model(load_base_only=options['base_model_only'])

            self.stdout.write('✅ Chatbot cargado correctamente')

            # Mostrar información del modelo
            model_info = "Modelo base" if options['base_model_only'] else "Modelo con LoRA fine-tuning"
            self.stdout.write(f'🏷️  Usando: {model_info}')

            # Obtener estadísticas de embeddings
            try:
                stats = chatbot.embedding_manager.get_stats()
                self.stdout.write(f'📊 Productos en índice: {stats.get("total_products", "N/A")}')
                self.stdout.write(f'🏷️  Categorías disponibles: {len(stats.get("categories", {}))}')
            except:
                self.stdout.write('⚠️ No se pudieron cargar estadísticas de productos')

            # Modo de operación
            if options['interactive']:
                self._interactive_mode(chatbot, options['save_conversation'])
            elif options['test_queries']:
                self._test_specific_queries(chatbot, options['test_queries'])
            else:
                self._run_default_tests(chatbot)

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error: {str(e)}')
            )
            raise

    def _interactive_mode(self, chatbot, save_conversation):
        """Modo interactivo de conversación"""
        self.stdout.write(
            self.style.SUCCESS('\n💬 MODO INTERACTIVO ACTIVADO')
        )
        self.stdout.write('Escriba "salir" para terminar, "limpiar" para reiniciar conversación')
        self.stdout.write('=' * 60)

        try:
            while True:
                user_input = input('\n👤 Tú: ').strip()

                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break
                elif user_input.lower() in ['limpiar', 'clear', 'reset']:
                    chatbot.clear_conversation()
                    self.stdout.write('🧹 Conversación reiniciada')
                    continue
                elif user_input.lower() in ['stats', 'estadisticas']:
                    stats = chatbot.get_conversation_stats()
                    self.stdout.write(f'📊 Interacciones: {stats["total_interactions"]}')
                    self.stdout.write(f'🎯 Búsquedas exitosas: {stats["success_rate"]}')
                    continue

                if not user_input:
                    continue

                # Obtener respuesta del chatbot
                self.stdout.write('🤖 Procesando...', ending='')
                response = chatbot.chat(user_input)
                self.stdout.write('\r🤖 Bot: ' + response)

        except KeyboardInterrupt:
            self.stdout.write('\n\n👋 Conversación terminada por el usuario')

        # Mostrar estadísticas finales
        stats = chatbot.get_conversation_stats()
        if stats['total_interactions'] > 0:
            self.stdout.write(
                self.style.SUCCESS('\n📊 ESTADÍSTICAS DE LA CONVERSACIÓN:')
            )
            self.stdout.write(f'💬 Total interacciones: {stats["total_interactions"]}')
            self.stdout.write(f'🎯 Búsquedas exitosas: {stats["success_rate"]}')

            # Guardar conversación si se solicita
            if save_conversation:
                try:
                    filepath = chatbot.save_conversation()
                    self.stdout.write(f'💾 Conversación guardada en: {filepath}')
                except Exception as e:
                    self.stdout.write(f'⚠️ Error guardando conversación: {str(e)}')

    def _test_specific_queries(self, chatbot, queries):
        """Prueba consultas específicas"""
        self.stdout.write(
            self.style.SUCCESS(f'\n🧪 PROBANDO {len(queries)} CONSULTAS ESPECÍFICAS')
        )

        for i, query in enumerate(queries, 1):
            self.stdout.write(f'\n--- Prueba {i}/{len(queries)} ---')
            self.stdout.write(f'👤 Usuario: {query}')

            response = chatbot.chat(query)
            self.stdout.write(f'🤖 Bot: {response}')

    def _run_default_tests(self, chatbot):
        """Ejecuta pruebas predeterminadas"""
        test_cases = [
            # Saludos y presentación
            {
                'category': 'SALUDOS',
                'queries': [
                    'Hola',
                    '¿Qué productos tienes?',
                    '¿Cómo funciona esto?'
                ]
            },
            # Búsquedas por categoría
            {
                'category': 'BÚSQUEDAS POR CATEGORÍA',
                'queries': [
                    'Busco celulares',
                    'Necesito un computador',
                    '¿Tienes televisores?',
                    'Quiero audífonos'
                ]
            },
            # Búsquedas por marca
            {
                'category': 'BÚSQUEDAS POR MARCA',
                'queries': [
                    'Productos Samsung',
                    '¿Tienes algo de Apple?',
                    'Celulares Xiaomi',
                    'Computadores HP'
                ]
            },
            # Búsquedas de ofertas
            {
                'category': 'BÚSQUEDAS DE OFERTAS',
                'queries': [
                    '¿Hay ofertas?',
                    'Productos en descuento',
                    'Celulares baratos',
                    'Computadores en promoción'
                ]
            },
            # Búsquedas por precio
            {
                'category': 'BÚSQUEDAS POR PRECIO',
                'queries': [
                    'Celular menos de 500 mil',
                    'Computador entre 1 y 2 millones',
                    'Productos hasta 100 mil',
                    'Televisor máximo 1 millón'
                ]
            },
            # Búsquedas específicas
            {
                'category': 'BÚSQUEDAS ESPECÍFICAS',
                'queries': [
                    'Celular con buena cámara',
                    'Laptop para gaming',
                    'Audífonos inalámbricos',
                    'Smart TV 55 pulgadas'
                ]
            },
            # Consultas conversacionales
            {
                'category': 'CONSULTAS CONVERSACIONALES',
                'queries': [
                    '¿Cuál me recomiendas?',
                    'No me convence, ¿tienes otro?',
                    'Necesito algo más barato',
                    'Gracias por la ayuda'
                ]
            }
        ]

        self.stdout.write(
            self.style.SUCCESS('\n🧪 EJECUTANDO PRUEBAS PREDETERMINADAS')
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
                    self.stdout.write(f'🤖 Bot: {response}')

                    # Pausa breve para lectura
                    import time
                    time.sleep(1)

                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'❌ Error en consulta: {str(e)}')
                    )

        # Estadísticas finales
        stats = chatbot.get_conversation_stats()
        self.stdout.write(
            self.style.SUCCESS('\n📊 ESTADÍSTICAS DE PRUEBA:')
        )
        self.stdout.write(f'✅ Pruebas completadas: {total_tests}')
        self.stdout.write(f'💬 Total interacciones: {stats["total_interactions"]}')
        self.stdout.write(f'🎯 Búsquedas exitosas: {stats["success_rate"]}')

        # Evaluación básica de rendimiento
        self._evaluate_performance(chatbot)

    def _evaluate_performance(self, chatbot):
        """Evaluación básica de rendimiento del chatbot"""
        self.stdout.write(
            self.style.SUCCESS('\n🔍 EVALUACIÓN DE RENDIMIENTO:')
        )

        # Prueba de búsqueda de embeddings
        try:
            search_results = chatbot.embedding_manager.search_products("celular samsung", top_k=5)
            self.stdout.write(
                f'📱 Búsqueda embeddings: {"✅" if search_results else "❌"} ({len(search_results)} resultados)')
        except:
            self.stdout.write('📱 Búsqueda embeddings: ❌ Error')

        # Prueba de generación de respuesta
        try:
            test_response = chatbot._generate_response("Hola", "Productos disponibles: iPhone 12, Samsung Galaxy")
            response_quality = "✅" if len(test_response) > 20 and "hola" in test_response.lower() else "⚠️"
            self.stdout.write(f'💬 Generación de respuesta: {response_quality}')
        except:
            self.stdout.write('💬 Generación de respuesta: ❌ Error')

        # Estadísticas de productos
        try:
            stats = chatbot.embedding_manager.get_stats()
            products_ok = "✅" if stats.get('total_products', 0) > 100 else "⚠️"
            self.stdout.write(f'📦 Base de productos: {products_ok} ({stats.get("total_products", 0)} productos)')

            categories_ok = "✅" if len(stats.get('categories', {})) > 5 else "⚠️"
            self.stdout.write(f'🏷️  Categorías: {categories_ok} ({len(stats.get("categories", {}))} categorías)')
        except:
            self.stdout.write('📦 Base de productos: ❌ Error accediendo a estadísticas')

        # Recomendaciones
        self.stdout.write(
            self.style.SUCCESS('\n💡 RECOMENDACIONES:')
        )

        conversation_stats = chatbot.get_conversation_stats()
        success_rate = float(conversation_stats['success_rate'].replace('%', ''))

        if success_rate < 50:
            self.stdout.write('• Considere reentrenar el modelo con más datos')
            self.stdout.write('• Verifique que los embeddings estén correctamente creados')
        elif success_rate < 80:
            self.stdout.write('• El rendimiento es aceptable pero puede mejorarse')
            self.stdout.write('• Considere ajustar los parámetros de búsqueda')
        else:
            self.stdout.write('• ¡Excelente rendimiento!')
            self.stdout.write('• El chatbot está funcionando correctamente')

        self.stdout.write(f'• Para más pruebas use: python manage.py test_chatbot --interactive')
