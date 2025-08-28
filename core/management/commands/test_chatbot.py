from django.core.management.base import BaseCommand
from core.chatbot.TechChatbot import TechChatbot
import os


class Command(BaseCommand):
    help = 'Prueba interactiva del chatbot con Groq + Llama 3.1'

    def add_arguments(self, parser):
        parser.add_argument(
            '--interactive',
            action='store_true',
            help='Activa el modo de chat interactivo'
        )
        parser.add_argument(
            '--api-key',
            type=str,
            help='Tu API key de Groq (opcional, puede usarse env variable)'
        )

    def handle(self, *args, **options):
        # Obtener API key de argumento o environment variable
        api_key = options['api_key'] or os.getenv('GROQ_API_KEY')

        if not api_key:
            self.stdout.write(self.style.ERROR(
                "❌ No se encontró API key de Groq. Usa:\n"
                "   --api-key TU_API_KEY\n"
                "   o configura la variable de entorno GROQ_API_KEY"
            ))
            return

        self.stdout.write(self.style.SUCCESS("🚀 Inicializando chatbot con Groq + Llama 3.1..."))

        # Inicializar chatbot
        chatbot = TechChatbot(api_key)

        self.stdout.write(self.style.SUCCESS("✅ Chatbot inicializado. Listo para conversar!"))

        if options['interactive']:
            self._interactive_mode(chatbot)
        else:
            self._test_mode(chatbot)

    def _interactive_mode(self, chatbot):
        """Modo interactivo de chat"""
        self.stdout.write(self.style.WARNING(
            "\n💬 MODO INTERACTIVO - Escribe tu mensaje\n"
            "   Comandos especiales:\n"
            "   - 'clear' → Limpiar historial\n"
            "   - 'stats' → Ver estadísticas\n"
            "   - 'exit' → Salir\n"
            "   - 'products' → Buscar productos específicos\n"
        ))

        while True:
            try:
                user_input = input("\n👤 Tú: ").strip()

                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break
                elif user_input.lower() == 'clear':
                    chatbot.clear_history()
                    self.stdout.write(self.style.SUCCESS("🧹 Historial limpiado"))
                    continue
                elif user_input.lower() == 'stats':
                    stats = chatbot.get_chat_stats()
                    self.stdout.write(self.style.INFO(
                        f"📊 Estadísticas: {stats['total_messages']} mensajes, "
                        f"{stats['user_messages']} user, {stats['assistant_messages']} assistant"
                    ))
                    continue
                elif user_input.lower() == 'products':
                    self._product_search_mode(chatbot)
                    continue

                if not user_input:
                    continue

                # Obtener respuesta
                response = chatbot.chat(user_input)

                # Mostrar respuesta formateada
                self.stdout.write(self.style.SUCCESS(f"🤖 Asistente: {response}"))

            except KeyboardInterrupt:
                self.stdout.write(self.style.WARNING("\n\n🛑 Chat interrumpido"))
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Error: {e}"))

    def _test_mode(self, chatbot):
        """Modo de prueba automática"""
        self.stdout.write(self.style.INFO("🧪 Ejecutando pruebas automáticas...\n"))

        test_cases = [
            "hola",
            "¿Qué productos tecnológicos tienes?",
            "Busco una laptop para programar",
            "¿Tienes celulares Samsung?",
            "Muéstrame ofertas en televisores",
            "gracias, adiós"
        ]

        for i, test_input in enumerate(test_cases, 1):
            self.stdout.write(self.style.INFO(f"🔸 Prueba {i}: '{test_input}'"))

            try:
                response = chatbot.chat(test_input)
                self.stdout.write(self.style.SUCCESS(f"   🤖: {response}"))
                self.stdout.write("   " + "─" * 60)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"   ❌ Error: {e}"))

    def _product_search_mode(self, chatbot):
        """Modo especial para búsqueda de productos"""
        self.stdout.write(self.style.INFO(
            "\n🔍 MODO BÚSQUEDA DE PRODUCTOS\n"
            "   Ejemplos: 'laptop i7', 'samsung galaxy', 'tv 55 pulgadas'"
        ))

        while True:
            try:
                search_query = input("\n🔎 Búsqueda: ").strip()

                if search_query.lower() in ['volver', 'back', 'exit']:
                    break
                if not search_query:
                    continue

                # Buscar productos directamente
                products = chatbot.embedding_manager.search_products(search_query, top_k=5)

                if not products:
                    self.stdout.write(self.style.WARNING("   ⚠️ No se encontraron productos"))
                    continue

                self.stdout.write(self.style.SUCCESS(f"   ✅ Encontrados {len(products)} productos:"))

                for i, product in enumerate(products, 1):
                    self.stdout.write(self.style.INFO(
                        f"   {i}. {product.get('name', 'Sin nombre')}\n"
                        f"      💰 ${product.get('price', 0):,.0f} | "
                        f"🏷️ {product.get('brand', 'Sin marca')}\n"
                        f"      📦 {product.get('category', 'Sin categoría')}\n"
                        f"      ⭐ Similitud: {product.get('similarity_score', 0):.2f}"
                    ))

                    # Mostrar specs importantes
                    specs = product.get('specifications', {})
                    if specs:
                        important_specs = []
                        for key in ['RAM', 'Almacenamiento', 'Procesador', 'Pantalla']:
                            if key in specs:
                                important_specs.append(f"{key}: {specs[key]}")
                        if important_specs:
                            self.stdout.write(f"      ⚙️ {', '.join(important_specs)}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"   ❌ Error en búsqueda: {e}"))