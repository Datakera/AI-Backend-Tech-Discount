from django.core.management.base import BaseCommand
from core.chatbot.TechChatbot import TechChatbot
from core.chatbot.EmbeddingManager import EmbeddingManager
import time
import json
import os
from datetime import datetime
import torch
import logging

logging.basicConfig(level=logging.INFO)


class Command(BaseCommand):
    help = 'Realiza benchmark completo del sistema de chatbot'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output-file',
            type=str,
            help='Archivo donde guardar los resultados del benchmark'
        )
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Incluye métricas detalladas en el benchmark'
        )
        parser.add_argument(
            '--test-embeddings',
            action='store_true',
            help='Incluye pruebas de rendimiento de embeddings'
        )

    def handle(self, *args, **options):
        start_time = datetime.now()

        self.stdout.write(
            self.style.SUCCESS('⚡ INICIANDO BENCHMARK DEL CHATBOT')
        )

        results = {
            'benchmark_date': start_time.isoformat(),
            'system_info': self._get_system_info(),
            'tests': {}
        }

        try:
            # 1. Benchmark de Embeddings
            if options['test_embeddings']:
                self.stdout.write('\n📊 Benchmarking Embeddings...')
                results['tests']['embeddings'] = self._benchmark_embeddings()

            # 2. Benchmark de Chatbot
            self.stdout.write('\n🤖 Benchmarking Chatbot...')
            results['tests']['chatbot'] = self._benchmark_chatbot()

            # 3. Benchmark de Búsqueda
            self.stdout.write('\n🔍 Benchmarking Sistema de Búsqueda...')
            results['tests']['search'] = self._benchmark_search()

            # 4. Pruebas de Estrés
            self.stdout.write('\n💪 Pruebas de Estrés...')
            results['tests']['stress'] = self._stress_tests()

            # Calcular métricas generales
            end_time = datetime.now()
            results['total_duration'] = (end_time - start_time).total_seconds()
            results['summary'] = self._generate_summary(results)

            # Mostrar resultados
            self._display_results(results)

            # Guardar resultados
            if options['output_file']:
                self._save_results(results, options['output_file'])

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error durante benchmark: {str(e)}')
            )
            raise

    def _get_system_info(self):
        """Obtiene información del sistema"""
        info = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

        return info

    def _benchmark_embeddings(self):
        """Benchmark del sistema de embeddings"""
        try:
            embedding_manager = EmbeddingManager()

            # Pruebas de búsqueda
            search_queries = [
                "celular samsung",
                "computador gaming",
                "audífonos inalámbricos",
                "televisor 55 pulgadas",
                "laptop para trabajo"
            ]

            search_times = []
            search_results_count = []

            for query in search_queries:
                start = time.time()
                results = embedding_manager.search_products(query, top_k=10)
                end = time.time()

                search_times.append(end - start)
                search_results_count.append(len(results))

            # Estadísticas
            stats = embedding_manager.get_stats()

            return {
                'total_products_indexed': stats.get('total_products', 0),
                'total_categories': len(stats.get('categories', {})),
                'avg_search_time_ms': sum(search_times) / len(search_times) * 1000,
                'min_search_time_ms': min(search_times) * 1000,
                'max_search_time_ms': max(search_times) * 1000,
                'avg_results_per_search': sum(search_results_count) / len(search_results_count),
                'search_success_rate': sum(1 for count in search_results_count if count > 0) / len(
                    search_results_count) * 100
            }

        except Exception as e:
            return {'error': str(e)}

    def _benchmark_chatbot(self):
        """Benchmark del chatbot completo"""
        try:
            chatbot = TechChatbot()

            # Intentar cargar modelo entrenado, sino usar base
            try:
                chatbot.load_model()
                model_type = "fine_tuned"
            except:
                chatbot.load_model(load_base_only=True)
                model_type = "base_model"

            # Consultas de prueba
            test_queries = [
                "Hola, ¿qué productos tienes?",
                "Busco celulares Samsung",
                "¿Hay computadores en oferta?",
                "Necesito audífonos baratos",
                "¿Cuál TV me recomiendas?",
                "Productos entre 200 y 500 mil pesos"
            ]

            response_times = []
            response_lengths = []
            successful_searches = 0

            for query in test_queries:
                start = time.time()
                response = chatbot.chat(query)
                end = time.time()

                response_times.append(end - start)
                response_lengths.append(len(response))

                # Verificar si la búsqueda fue exitosa (simplificado)
                if len(response) > 50 and any(
                        word in response.lower() for word in ['producto', 'precio', '$', 'disponible']):
                    successful_searches += 1

            # Estadísticas de conversación
            conv_stats = chatbot.get_conversation_stats()

            return {
                'model_type': model_type,
                'total_queries_tested': len(test_queries),
                'avg_response_time_s': sum(response_times) / len(response_times),
                'min_response_time_s': min(response_times),
                'max_response_time_s': max(response_times),
                'avg_response_length': sum(response_lengths) / len(response_lengths),
                'successful_searches': successful_searches,
                'success_rate_percent': successful_searches / len(test_queries) * 100,
                'conversation_stats': conv_stats
            }

        except Exception as e:
            return {'error': str(e)}

    def _benchmark_search(self):
        """Benchmark del sistema de búsqueda combinado"""
        try:
            chatbot = TechChatbot()
            try:
                chatbot.load_model()
            except:
                chatbot.load_model(load_base_only=True)

            # Casos de prueba de búsqueda
            search_cases = [
                {"query": "Samsung Galaxy", "expected_type": "celular"},
                {"query": "laptop gamer", "expected_type": "computador"},
                {"query": "productos en descuento", "expected_type": "ofertas"},
                {"query": "menos de 300 mil", "expected_type": "precio"},
                {"query": "audífonos Sony", "expected_type": "marca_especifica"}
            ]

            intent_accuracies = []
            search_effectiveness = []

            for case in search_cases:
                # Extraer intención
                intent = chatbot._extract_search_intent(case["query"])

                # Verificar precisión de la intención (simplificado)
                intent_correct = False
                if case["expected_type"] == "celular" and any(
                        cat in str(intent.get('category', '')) for cat in ['celular']):
                    intent_correct = True
                elif case["expected_type"] == "computador" and any(
                        cat in str(intent.get('category', '')) for cat in ['computador']):
                    intent_correct = True
                elif case["expected_type"] == "ofertas" and intent.get('looking_for_deals'):
                    intent_correct = True
                elif case["expected_type"] == "precio" and (
                        intent['price_range']['min'] or intent['price_range']['max']):
                    intent_correct = True
                elif case["expected_type"] == "marca_especifica" and intent.get('brand'):
                    intent_correct = True

                intent_accuracies.append(intent_correct)

                # Buscar productos
                products = chatbot._search_products(intent, top_k=5)
                search_effectiveness.append(len(products) > 0)

            return {
                'intent_extraction_accuracy': sum(intent_accuracies) / len(intent_accuracies) * 100,
                'search_effectiveness': sum(search_effectiveness) / len(search_effectiveness) * 100,
                'total_test_cases': len(search_cases)
            }

        except Exception as e:
            return {'error': str(e)}

    def _stress_tests(self):
        """Pruebas de estrés del sistema"""
        try:
            chatbot = TechChatbot()
            try:
                chatbot.load_model()
            except:
                chatbot.load_model(load_base_only=True)

            # Prueba de múltiples consultas consecutivas
            stress_queries = ["busco celulares"] * 20  # 20 consultas idénticas

            start_time = time.time()
            response_times = []

            for query in stress_queries:
                query_start = time.time()
                chatbot.chat(query)
                query_end = time.time()
                response_times.append(query_end - query_start)

            end_time = time.time()

            # Prueba de memoria
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / 1024 / 1024

            return {
                'consecutive_queries': len(stress_queries),
                'total_time_s': end_time - start_time,
                'avg_time_per_query_s': sum(response_times) / len(response_times),
                'memory_usage_mb': memory_usage_mb,
                'performance_degradation': (max(response_times) - min(response_times)) / min(response_times) * 100
            }

        except Exception as e:
            return {'error': str(e)}

    def _generate_summary(self, results):
        """Genera resumen de los resultados"""
        summary = {
            'overall_status': 'good',
            'critical_issues': [],
            'recommendations': []
        }

        # Analizar resultados de embeddings
        if 'embeddings' in results['tests']:
            emb = results['tests']['embeddings']
            if emb.get('avg_search_time_ms', 1000) > 500:
                summary['recommendations'].append('Considere optimizar el índice de embeddings')
            if emb.get('search_success_rate', 0) < 80:
                summary['critical_issues'].append('Baja tasa de éxito en búsquedas de embeddings')

        # Analizar resultados del chatbot
        if 'chatbot' in results['tests']:
            chat = results['tests']['chatbot']
            if chat.get('avg_response_time_s', 10) > 5:
                summary['recommendations'].append('Tiempo de respuesta alto, considere optimización')
            if chat.get('success_rate_percent', 0) < 70:
                summary['critical_issues'].append('Baja tasa de éxito del chatbot')

        # Determinar estado general
        if len(summary['critical_issues']) > 2:
            summary['overall_status'] = 'critical'
        elif len(summary['critical_issues']) > 0:
            summary['overall_status'] = 'warning'

        return summary

    def _display_results(self, results):
        """Muestra los resultados del benchmark"""
        self.stdout.write(
            self.style.SUCCESS('\n📋 RESULTADOS DEL BENCHMARK')
        )
        self.stdout.write('=' * 60)

        # Información del sistema
        sys_info = results['system_info']
        self.stdout.write(f'🖥️  Sistema: Python {sys_info["python_version"]}, PyTorch {sys_info["torch_version"]}')
        if sys_info['cuda_available']:
            self.stdout.write(f'🔥 GPU: {sys_info["gpu_name"]} ({sys_info["gpu_memory_gb"]:.1f}GB)')
        else:
            self.stdout.write('💻 CPU: Sin GPU detectada')

        # Resultados de pruebas
        for test_name, test_results in results['tests'].items():
            if 'error' in test_results:
                self.stdout.write(f'\n❌ {test_name.upper()}: Error - {test_results["error"]}')
                continue

            self.stdout.write(f'\n✅ {test_name.upper()}:')

            if test_name == 'embeddings':
                self.stdout.write(f'  📦 Productos indexados: {test_results["total_products_indexed"]:,}')
                self.stdout.write(f'  🏷️  Categorías: {test_results["total_categories"]}')
                self.stdout.write(f'  ⚡ Búsqueda promedio: {test_results["avg_search_time_ms"]:.1f}ms')
                self.stdout.write(f'  🎯 Tasa de éxito: {test_results["search_success_rate"]:.1f}%')

            elif test_name == 'chatbot':
                self.stdout.write(f'  🤖 Modelo: {test_results["model_type"]}')
                self.stdout.write(f'  ⚡ Respuesta promedio: {test_results["avg_response_time_s"]:.2f}s')
                self.stdout.write(f'  📝 Longitud promedio: {test_results["avg_response_length"]} caracteres')
                self.stdout.write(f'  🎯 Tasa de éxito: {test_results["success_rate_percent"]:.1f}%')

            elif test_name == 'search':
                self.stdout.write(f'  🧠 Precisión de intención: {test_results["intent_extraction_accuracy"]:.1f}%')
                self.stdout.write(f'  🔍 Efectividad de búsqueda: {test_results["search_effectiveness"]:.1f}%')

            elif test_name == 'stress':
                self.stdout.write(f'  💪 Consultas consecutivas: {test_results["consecutive_queries"]}')
                self.stdout.write(f'  ⏱️  Tiempo total: {test_results["total_time_s"]:.2f}s')
                self.stdout.write(f'  🧠 Memoria usada: {test_results["memory_usage_mb"]:.1f}MB')
                self.stdout.write(f'  📉 Degradación: {test_results["performance_degradation"]:.1f}%')

        # Resumen
        summary = results['summary']
        status_icon = {'good': '✅', 'warning': '⚠️', 'critical': '❌'}[summary['overall_status']]
        self.stdout.write(f'\n{status_icon} ESTADO GENERAL: {summary["overall_status"].upper()}')

        if summary['critical_issues']:
            self.stdout.write('\n🚨 PROBLEMAS CRÍTICOS:')
            for issue in summary['critical_issues']:
                self.stdout.write(f'  • {issue}')

        if summary['recommendations']:
            self.stdout.write('\n💡 RECOMENDACIONES:')
            for rec in summary['recommendations']:
                self.stdout.write(f'  • {rec}')

        self.stdout.write(f'\n⏱️  Duración total del benchmark: {results["total_duration"]:.2f}s')

    def _save_results(self, results, filename):
        """Guarda los resultados en un archivo JSON"""
        try:
            # Crear directorio si no existe
            output_dir = "data/benchmarks"
            os.makedirs(output_dir, exist_ok=True)

            # Si no se especifica extensión, agregar .json
            if not filename.endswith('.json'):
                filename += '.json'

            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.stdout.write(f'\n💾 Resultados guardados en: {filepath}')

        except Exception as e:
            self.stdout.write(f'⚠️ Error guardando resultados: {str(e)}')
