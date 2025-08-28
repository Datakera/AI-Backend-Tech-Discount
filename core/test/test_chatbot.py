from django.test import TestCase
from unittest.mock import Mock, patch, MagicMock
from core.chatbot.ChatbotTrainer import ChatbotTrainer
from datasets import Dataset
import logging

# Configurar logging para pruebas
logging.basicConfig(level=logging.ERROR)


class TestChatbotTrainer(TestCase):

    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.trainer = ChatbotTrainer(model_name="microsoft/DialoGPT-small")

    def test_create_training_dataset_with_valid_products(self):
        """Test con productos válidos de MongoDB"""
        # Mock de productos válidos
        mock_products = [
            {
                'name': 'iPhone 13',
                'brand': 'Apple',
                'discount_price_num': 999,
                'original_price_num': 1099,
                'discount_percent': '10%',
                'category': 'celulares'
            },
            {
                'name': 'Galaxy S21',
                'brand': 'Samsung',
                'discount_price_num': 899,
                'original_price_num': 999,
                'discount_percent': '15%',
                'category': 'celulares'
            }
        ]

        with patch('core.chatbot.ChatbotTrainer.MongoManager') as mock_mongo:
            mock_instance = mock_mongo.return_value
            mock_instance.get_all_products.return_value = mock_products

            # Ejecutar la función
            dataset = self.trainer.create_training_dataset(max_samples=10)

            # Verificaciones
            self.assertIsInstance(dataset, Dataset)
            self.assertGreater(len(dataset), 0)
            self.assertIn('text', dataset.features)

    def test_create_training_dataset_with_string_error(self):
        """Test cuando MongoDB devuelve un string (error)"""
        with patch('core.chatbot.ChatbotTrainer.MongoManager') as mock_mongo:
            mock_instance = mock_mongo.return_value
            mock_instance.get_all_products.return_value = "Error de conexión a MongoDB"

            # Ejecutar la función - debería usar fallback
            dataset = self.trainer.create_training_dataset(max_samples=10)

            # Verificar que se usa el dataset de fallback
            self.assertIsInstance(dataset, Dataset)
            self.assertGreater(len(dataset), 0)

    def test_create_training_dataset_with_empty_list(self):
        """Test cuando MongoDB devuelve lista vacía"""
        with patch('core.chatbot.ChatbotTrainer.MongoManager') as mock_mongo:
            mock_instance = mock_mongo.return_value
            mock_instance.get_all_products.return_value = []

            # Ejecutar la función - debería usar fallback
            dataset = self.trainer.create_training_dataset(max_samples=10)

            # Verificar que se usa el dataset de fallback
            self.assertIsInstance(dataset, Dataset)
            self.assertGreater(len(dataset), 0)

    def test_create_training_dataset_with_invalid_products(self):
        """Test cuando MongoDB devuelve productos inválidos"""
        invalid_products = ["producto1", "producto2", 123]  # No son diccionarios

        with patch('core.chatbot.ChatbotTrainer.MongoManager') as mock_mongo:
            mock_instance = mock_mongo.return_value
            mock_instance.get_all_products.return_value = invalid_products

            # Ejecutar la función - debería usar fallback
            dataset = self.trainer.create_training_dataset(max_samples=10)

            # Verificar que se usa el dataset de fallback
            self.assertIsInstance(dataset, Dataset)
            self.assertGreater(len(dataset), 0)

    def test_create_training_dataset_with_mixed_products(self):
        """Test cuando MongoDB devuelve mix de productos válidos e inválidos"""
        mixed_products = [
            {'name': 'iPhone', 'brand': 'Apple', 'price': 999},  # Válido
            "invalid_product_string",  # Inválido
            123,  # Inválido
            {'name': 'Galaxy', 'brand': 'Samsung', 'price': 899}  # Válido
        ]

        with patch('core.chatbot.ChatbotTrainer.MongoManager') as mock_mongo:
            mock_instance = mock_mongo.return_value
            mock_instance.get_all_products.return_value = mixed_products

            # Ejecutar la función
            dataset = self.trainer.create_training_dataset(max_samples=10)

            # Verificar que funciona y crea dataset
            self.assertIsInstance(dataset, Dataset)

    def test_create_simple_conversations_valid_product(self):
        """Test de _create_simple_conversations con producto válido"""
        product = {
            'name': 'iPhone 13 Pro',
            'brand': 'Apple',
            'discount_price_num': 1199,
            'original_price_num': 1299,
            'discount_percent': '8%',
            'category': 'celulares'
        }

        conversations = self.trainer._create_simple_conversations(product)

        # Verificar que se crean conversaciones
        self.assertIsInstance(conversations, list)
        self.assertGreater(len(conversations), 0)

        # Verificar formato de las conversaciones
        for conv in conversations:
            self.assertIn('user', conv)
            self.assertIn('assistant', conv)

    def test_create_simple_conversations_invalid_product(self):
        """Test de _create_simple_conversations con producto inválido"""
        invalid_product = "not a dictionary"

        conversations = self.trainer._create_simple_conversations(invalid_product)

        # Verificar que devuelve lista vacía
        self.assertEqual(conversations, [])

    def test_format_conversation_for_model(self):
        """Test de _format_conversation_for_model"""
        conversation = {
            'user': 'hola',
            'assistant': 'Hola! ¿Cómo estás?'
        }

        formatted = self.trainer._format_conversation_for_model(conversation)

        # Verificar que devuelve string no vacío
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)

    def test_fallback_dataset_creation(self):
        """Test de _create_fallback_dataset"""
        fallback_dataset = self.trainer._create_fallback_dataset()

        # Verificar que crea dataset de fallback
        self.assertIsInstance(fallback_dataset, Dataset)
        self.assertGreater(len(fallback_dataset), 0)
        self.assertIn('text', fallback_dataset.features)

    def test_tokenize_function(self):
        """Test de tokenize_function"""
        # Primero necesitamos cargar el tokenizer
        self.trainer.load_model()

        examples = {
            'text': ['Usuario: hola\nAsistente: Hola!', 'Usuario: productos\nAsistente: Tengo varios productos']
        }

        tokenized = self.trainer.tokenize_function(examples)

        # Verificar que tokeniza correctamente
        self.assertIn('input_ids', tokenized)
        self.assertIn('attention_mask', tokenized)

    @patch('core.chatbot.ChatbotTrainer.Trainer')
    @patch('core.chatbot.ChatbotTrainer.Dataset')
    def test_train_method(self, mock_dataset, mock_trainer):
        """Test del método train"""
        # Configurar mocks
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance

        # Ejecutar train
        self.trainer.load_model()
        self.trainer.train(mock_dataset_instance, epochs=1, batch_size=2)

        # Verificar que se llaman los métodos esperados
        mock_trainer.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()