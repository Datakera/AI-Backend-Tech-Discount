# core/chatbot/utils/test_chatbot.py
from core.chatbot.bot import TechDiscountChatbot

def test_chatbot():
    chatbot = TechDiscountChatbot()

    test_questions = [
        "Busco un computador con 16GB de RAM y SSD",
        "Recomiéndame smartphones con buen descuento",
        "Necesito una laptop para programar que esté en oferta",
        "¿Qué televisores 4K tienen descuento?",
        "Busco tablets con al menos 128GB de almacenamiento"
    ]

    for question in test_questions:
        print(f"\n🧑‍💻 Usuario: {question}")
        response, sources = chatbot.ask(question)
        print(f"🤖 Chatbot: {response}")
        print(f"📚 Fuentes: {len(sources)} productos encontrados")

        for i, source in enumerate(sources, 1):
            print(f"   {i}. {source.metadata['name'][:50]}...")


if __name__ == "__main__":
    test_chatbot()