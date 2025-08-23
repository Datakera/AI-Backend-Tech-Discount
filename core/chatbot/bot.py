# core/chatbot/bot.py - Limpio y funcional
from .vector_store import ProductVectorStore
from .chains.base import BaseChainFactory


class TechDiscountChatbot:
    def __init__(self):
        self.vector_store = ProductVectorStore()
        self.recommendation_chain = None
        self.comparison_chain = None
        self.setup_chains()

    def setup_chains(self):
        """Configura las cadenas de procesamiento"""
        # Cargar vector store
        if not self.vector_store.load_vector_store():
            self.vector_store.build_vector_store()

        # Inicializar LLM
        from langchain_openai import OpenAI
        from django.conf import settings

        llm = OpenAI(
            temperature=0.1,
            max_tokens=500,
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Crear cadenas
        self.recommendation_chain = BaseChainFactory.create_product_recommendation_chain(
            self.vector_store.vector_store
        )

        self.comparison_chain = BaseChainFactory.create_product_comparison_chain(
            self.vector_store.vector_store, llm
        )

        print("✅ Cadenas configuradas correctamente")

    def ask_recommendation(self, question):
        """Para preguntas de recomendación"""
        try:
            result = self.recommendation_chain({"query": question})
            return self._format_response(result)
        except Exception as e:
            return self._format_error(e)

    def ask_comparison(self, question):
        """Para preguntas de comparación"""
        try:
            result = self.comparison_chain({"query": question})
            return self._format_response(result)
        except Exception as e:
            return self._format_error(e)

    def _format_response(self, result):
        """Formatea la respuesta"""
        try:
            answer = result.get("result") or result.get("output") or "No se pudo generar respuesta"
            sources = result.get("source_documents", [])

            return {
                "answer": answer,
                "sources": [
                    {
                        "name": doc.metadata.get("name", "Sin nombre"),
                        "brand": doc.metadata.get("brand", "Sin marca"),
                        "price": doc.metadata.get("price", 0),
                        "discount": doc.metadata.get("discount", "0%"),
                        "url": doc.metadata.get("url", "#")
                    }
                    for doc in sources
                ],
                "total_products": len(sources)
            }
        except Exception as e:
            return self._format_error(e)

    def _format_error(self, error):
        """Formatea errores"""
        return {
            "answer": f"⚠️ Error: {str(error)}",
            "sources": [],
            "total_products": 0
        }

    def ask(self, question):
        """Método principal"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["compar", "vs", "diferencia"]):
            return self.ask_comparison(question)
        else:
            return self.ask_recommendation(question)