# core/chatbot/chains/base.py - Versión corregida
from langchain.chains import RetrievalQA
from langchain_openai  import OpenAI
from langchain_core.prompts import PromptTemplate
from django.conf import settings


class BaseChainFactory:
    @staticmethod
    def create_product_recommendation_chain(vector_store):
        """Crea cadena de recomendación de productos - VERSIÓN CORREGIDA"""
        # Configurar el prompt
        prompt_template = """
        Eres un asistente especializado en recomendar productos tecnológicos con descuento.
        Basa tus respuestas únicamente en los productos proporcionados en el contexto.

        Contexto:
        {context}

        Pregunta: {question}

        Responde en español recomendando los mejores productos encontrados.
        Incluye precio, descuento y características relevantes.
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Configurar LLM
        llm = OpenAI(
            temperature=0.1,
            max_tokens=500,
            openai_api_key=settings.OPENAI_API_KEY
        )

        # Crear cadena con formato consistente
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="result"  # ← Asegurar output key consistente
        )

    @staticmethod
    def create_product_comparison_chain(vector_store, llm=None):
        """Crea cadena de comparación de productos"""
        # Configurar LLM si no se proporciona
        if llm is None:
            from langchain_openai import OpenAI
            from django.conf import settings
            llm = OpenAI(
                temperature=0.1,
                max_tokens=600,
                openai_api_key=settings.OPENAI_API_KEY
            )

        # Configurar el prompt de comparación
        prompt_template = """
        Eres un experto en comparar productos tecnológicos. 
        Compara estos productos basándote en el contexto:

        Contexto:
        {context}

        Pregunta: {question}

        Proporciona una comparación detallada incluyendo:
        - Especificaciones técnicas
        - Relación precio-calidad  
        - Descuentos disponibles
        - Mejor opción por categoría
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Crear cadena
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="result"
        )

    @staticmethod
    def create_product_search_chain(vector_store, llm=None):
        """Crea cadena para búsquedas específicas"""
        if llm is None:
            from langchain_openai import OpenAI
            from django.conf import settings
            llm = OpenAI(
                temperature=0.1,
                max_tokens=400,
                openai_api_key=settings.OPENAI_API_KEY
            )

        prompt_template = """
        Eres un asistente para búsquedas específicas de productos tecnológicos.

        Contexto:
        {context}

        Pregunta: {question}

        Enfócate en encontrar productos que coincidan exactamente con los criterios solicitados.
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="result"
        )