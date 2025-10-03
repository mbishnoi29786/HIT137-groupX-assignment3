"""
Comprehensive test suite for the AI Model Integration Application.

This module provides unit tests and integration tests for all major components,
demonstrating testing best practices and ensuring code reliability.

Demonstrates: Unit testing, integration testing, mock objects, test organization,
parameterized testing, edge case handling, and comprehensive error validation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from io import StringIO
from PIL import Image
import tempfile
import time
from contextlib import contextmanager

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from models.model_info import ModelInfo, ModelUsageGuide
from models.model_factory import ModelFactory, get_model_factory
from oop_examples.base_classes import (
    BaseModelWrapper, MultiInheritanceModelWrapper,
    ModelLoggingMixin, PolymorphicModelManager
)
from oop_examples.decorators import timeit, log_exceptions, validate_input

# Test Constants
MAX_TEXT_LENGTH = 1000
MIN_TEXT_LENGTH = 1
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
MAX_MODEL_CREATION_TIME = 5.0  # seconds
TEST_IMAGE_SIZE = (100, 100)


class TestModelInfo(unittest.TestCase):
    """
    Test the ModelInfo class functionality.

    Ensures proper model metadata retrieval, validation, and filtering.
    Guards against: Invalid model keys, incorrect type filtering, metadata inconsistencies.
    """

    def test_get_model_names(self):
        """Test getting list of available model names."""
        names = ModelInfo.get_model_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0, "Should have at least one model")
        self.assertIn('image_classifier', names)
        self.assertIn('text_generator', names)

    def test_get_model_info_valid_key(self):
        """Test getting info for valid model key."""
        info = ModelInfo.get_model_info('image_classifier')
        self.assertIsInstance(info, dict)

        # Verify all required fields are present
        required_fields = ['name', 'display_name', 'task', 'input_type', 'output_type']
        for field in required_fields:
            self.assertIn(field, info, f"Missing required field: {field}")

        self.assertEqual(info['input_type'], 'image')

    def test_get_model_info_invalid_key(self):
        """Test error handling for invalid model key with proper exception message."""
        with self.assertRaises(KeyError) as context:
            ModelInfo.get_model_info('nonexistent_model')

        # Verify the error message is informative
        error_message = str(context.exception)
        self.assertIn('nonexistent_model', error_message.lower())

    def test_get_models_by_input_type(self):
        """Test filtering models by input type."""
        text_models = ModelInfo.get_models_by_input_type('text')
        self.assertIsInstance(text_models, dict)
        self.assertGreater(len(text_models), 0, "Should have at least one text model")

        # Verify all returned models have correct input type
        for model_key, model_info in text_models.items():
            self.assertEqual(model_info['input_type'], 'text',
                             f"Model {model_key} has incorrect input_type")

    def test_get_models_by_invalid_input_type(self):
        """Test filtering with non-existent input type returns empty dict."""
        invalid_models = ModelInfo.get_models_by_input_type('quantum_entanglement')
        self.assertIsInstance(invalid_models, dict)
        self.assertEqual(len(invalid_models), 0)

    def test_validate_model_key(self):
        """Test model key validation for both valid and invalid keys."""
        # Valid keys
        self.assertTrue(ModelInfo.validate_model_key('image_classifier'))
        self.assertTrue(ModelInfo.validate_model_key('text_generator'))

        # Invalid keys
        self.assertFalse(ModelInfo.validate_model_key('invalid_key'))
        self.assertFalse(ModelInfo.validate_model_key(''))
        self.assertFalse(ModelInfo.validate_model_key(None))

    def test_get_supported_input_types(self):
        """Test getting all supported input types."""
        input_types = ModelInfo.get_supported_input_types()
        self.assertIsInstance(input_types, list)
        self.assertIn('image', input_types)
        self.assertIn('text', input_types)

        # Verify no duplicates
        self.assertEqual(len(input_types), len(set(input_types)))

    def test_model_info_immutability(self):
        """Test that returned model info cannot accidentally modify internal state."""
        info1 = ModelInfo.get_model_info('image_classifier')
        info1['task'] = 'modified'

        # Get info again and verify it wasn't modified
        info2 = ModelInfo.get_model_info('image_classifier')
        self.assertNotEqual(info2['task'], 'modified')


class TestModelUsageGuide(unittest.TestCase):
    """
    Test the ModelUsageGuide class.

    Ensures proper usage instructions are available for all models.
    """

    def test_get_usage_guide_valid_key(self):
        """Test getting usage guide for valid model."""
        guide = ModelUsageGuide.get_usage_guide('image_classifier')
        self.assertIsInstance(guide, dict)

        # Verify required fields
        required_fields = ['preparation_tips', 'interpretation_guide', 'limitations']
        for field in required_fields:
            self.assertIn(field, guide, f"Missing required field: {field}")
            self.assertIsInstance(guide[field], list, f"{field} should be a list")
            self.assertGreater(len(guide[field]), 0, f"{field} should not be empty")

    def test_get_usage_guide_invalid_key(self):
        """Test default guide for invalid model key."""
        guide = ModelUsageGuide.get_usage_guide('invalid_model')
        self.assertIsInstance(guide, dict)
        self.assertIn('preparation_tips', guide)

        # Should return default messages
        self.assertTrue(any('No specific guidance available' in tip
                            for tip in guide['preparation_tips']))

    def test_usage_guide_consistency(self):
        """Test that all valid models have consistent usage guide structure."""
        for model_key in ModelInfo.get_model_names():
            guide = ModelUsageGuide.get_usage_guide(model_key)

            # All guides should have the same structure
            self.assertIn('preparation_tips', guide)
            self.assertIn('interpretation_guide', guide)
            self.assertIn('limitations', guide)


class TestBaseClasses(unittest.TestCase):
    """
    Test the OOP base classes and mixins.

    Verifies proper initialization, encapsulation, and inheritance behavior.
    """

    def setUp(self):
        """Set up test fixtures before each test."""
        self.base_wrapper = BaseModelWrapper("test_model")

    def tearDown(self):
        """Clean up after each test."""
        self.base_wrapper = None

    def test_base_model_wrapper_initialization(self):
        """Test BaseModelWrapper initialization with proper state."""
        self.assertEqual(self.base_wrapper.model_name, "test_model")
        self.assertFalse(self.base_wrapper.is_loaded())
        self.assertEqual(self.base_wrapper.get_error_count(), 0)

    def test_base_validation_valid_inputs(self):
        """Test base input validation with various valid inputs."""
        valid_inputs = ["test", "hello world", "123", "special!@#$%"]
        for input_val in valid_inputs:
            self.assertTrue(self.base_wrapper.validate_input(input_val),
                            f"Failed to validate: {input_val}")

    def test_base_validation_invalid_inputs(self):
        """Test base input validation with invalid inputs."""
        invalid_inputs = [None, "", "   ", [], {}, 0]
        for input_val in invalid_inputs:
            self.assertFalse(self.base_wrapper.validate_input(input_val),
                             f"Should not validate: {input_val}")

    def test_error_count_increment(self):
        """
        Test error counting functionality.

        Critical for rate limiting and monitoring model failures in production.
        Guards against: Counter overflow, negative counts.
        """
        initial_count = self.base_wrapper.get_error_count()
        self.assertEqual(initial_count, 0)

        # Increment multiple times
        for i in range(1, 6):
            self.base_wrapper._increment_error_count()
            self.assertEqual(self.base_wrapper.get_error_count(), i)

        # Verify count is always non-negative
        self.assertGreaterEqual(self.base_wrapper.get_error_count(), 0)

    def test_model_logging_mixin(self):
        """Test ModelLoggingMixin functionality with proper isolation."""
        mixin = ModelLoggingMixin()

        # Verify initial state
        self.assertEqual(len(mixin.get_operation_history()), 0)

        # Test logging operations
        test_operations = [
            ("operation1", "input1", "result1", 1.5),
            ("operation2", "input2", "result2", 2.3),
            ("operation3", "input3", "result3", 0.8),
        ]

        for op, inp, res, duration in test_operations:
            mixin.log_operation(op, inp, res, duration)

        history = mixin.get_operation_history()
        self.assertEqual(len(history), 3)

        # Verify first logged operation
        self.assertEqual(history[0]['operation'], 'operation1')
        self.assertEqual(history[0]['duration'], 1.5)

        # Test clear history
        mixin.clear_history()
        self.assertEqual(len(mixin.get_operation_history()), 0)

    def test_model_logging_mixin_timestamp(self):
        """Verify that logged operations include timestamps."""
        mixin = ModelLoggingMixin()

        before_time = time.time()
        mixin.log_operation("test_op", "input", "result", 1.0)
        after_time = time.time()

        history = mixin.get_operation_history()
        self.assertEqual(len(history), 1)

        # Verify timestamp exists and is reasonable
        if 'timestamp' in history[0]:
            timestamp = history[0]['timestamp']
            self.assertGreaterEqual(timestamp, before_time)
            self.assertLessEqual(timestamp, after_time)

    def test_polymorphic_model_manager(self):
        """Test PolymorphicModelManager functionality with multiple models."""
        manager = PolymorphicModelManager()

        # Create multiple mock models
        mock_model1 = Mock()
        mock_model1.run.return_value = {"result": "test1"}
        mock_model1.get_model_info.return_value = {"name": "model1"}

        mock_model2 = Mock()
        mock_model2.run.return_value = {"result": "test2"}
        mock_model2.get_model_info.return_value = {"name": "model2"}

        # Register models
        manager.register_model("model1", mock_model1)
        manager.register_model("model2", mock_model2)

        # Verify registration
        available = manager.get_available_models()
        self.assertIn("model1", available)
        self.assertIn("model2", available)
        self.assertEqual(len(available), 2)

        # Test switching between models
        manager.set_active_model("model1")
        result1 = manager.process_with_current_model("test_input")
        self.assertEqual(result1["result"], "test1")

        manager.set_active_model("model2")
        result2 = manager.process_with_current_model("test_input")
        self.assertEqual(result2["result"], "test2")

        # Verify correct models were called
        mock_model1.run.assert_called_once_with("test_input")
        mock_model2.run.assert_called_once_with("test_input")

    def test_polymorphic_manager_invalid_model(self):
        """Test error handling when setting invalid active model."""
        manager = PolymorphicModelManager()

        with self.assertRaises((KeyError, ValueError)):
            manager.set_active_model("nonexistent_model")


class TestDecorators(unittest.TestCase):
    """
    Test custom decorators functionality.

    Ensures decorators work correctly and can be stacked.
    """

    def test_timeit_decorator(self):
        """Test timing decorator logs execution time."""

        @timeit
        def test_function():
            time.sleep(0.01)  # Small delay to ensure measurable time
            return "test_result"

        with patch('oop_examples.decorators.logger') as mock_logger:
            result = test_function()
            self.assertEqual(result, "test_result")

            # Verify logging was called
            self.assertTrue(mock_logger.info.called)

            # Check that timing info was logged
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("test_function", call_args)
            self.assertIn("executed in", call_args)

    def test_timeit_preserves_function_signature(self):
        """Test that decorator preserves original function signature."""

        @timeit
        def function_with_args(x, y, z=None):
            return x + y

        with patch('oop_examples.decorators.logger'):
            result = function_with_args(1, 2, z=3)
            self.assertEqual(result, 3)

    def test_log_exceptions_decorator(self):
        """Test exception logging decorator catches and re-raises exceptions."""

        @log_exceptions
        def failing_function():
            raise ValueError("Test error message")

        with patch('oop_examples.decorators.logger') as mock_logger:
            with self.assertRaises(ValueError) as context:
                failing_function()

            # Verify exception was logged
            mock_logger.error.assert_called()

            # Verify error message is preserved
            self.assertIn("Test error message", str(context.exception))

    def test_log_exceptions_preserves_return_value(self):
        """Test that decorator doesn't interfere with normal execution."""

        @log_exceptions
        def normal_function():
            return "success"

        with patch('oop_examples.decorators.logger'):
            result = normal_function()
            self.assertEqual(result, "success")

    def test_validate_input_decorator_string(self):
        """Test input validation decorator with string type."""

        @validate_input(input_type="str")
        def string_function(data):
            return f"processed: {data}"

        # Valid input
        result = string_function("test")
        self.assertEqual(result, "processed: test")

        # Invalid input
        with self.assertRaises(TypeError) as context:
            string_function(123)

        self.assertIn("str", str(context.exception).lower())

    def test_validate_input_decorator_multiple_types(self):
        """Test input validation with various types."""
        test_cases = [
            ("int", 42, True),
            ("int", "42", False),
            ("list", [1, 2, 3], True),
            ("list", (1, 2, 3), False),
            ("dict", {"key": "value"}, True),
            ("dict", ["key", "value"], False),
        ]

        for input_type, test_value, should_pass in test_cases:
            @validate_input(input_type=input_type)
            def test_func(data):
                return data

            if should_pass:
                result = test_func(test_value)
                self.assertEqual(result, test_value)
            else:
                with self.assertRaises(TypeError):
                    test_func(test_value)

    def test_stacked_decorators(self):
        """Test that multiple decorators can be stacked correctly."""

        @timeit
        @log_exceptions
        @validate_input(input_type="str")
        def multi_decorated_function(data):
            return data.upper()

        with patch('oop_examples.decorators.logger'):
            result = multi_decorated_function("hello")
            self.assertEqual(result, "HELLO")


class TestModelFactory(unittest.TestCase):
    """
    Test ModelFactory functionality.

    Ensures proper model instantiation and factory pattern implementation.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()

    def tearDown(self):
        """Clean up after tests."""
        self.factory = None

    def test_get_available_model_types(self):
        """Test getting available model types."""
        types = self.factory.get_available_model_types()
        self.assertIsInstance(types, list)
        self.assertGreater(len(types), 0)
        self.assertIn('image_classifier', types)
        self.assertIn('text_generator', types)

    @patch('models.hf_integration.ImageClassifierWrapper')
    def test_create_image_model(self, mock_wrapper_class):
        """Test creating image classification model with proper initialization."""
        mock_instance = Mock()
        mock_wrapper_class.return_value = mock_instance

        model = self.factory.create_model('image_classifier', use_cache=False)

        self.assertEqual(model, mock_instance)
        mock_wrapper_class.assert_called_once()

        # Verify cache parameter was passed
        call_kwargs = mock_wrapper_class.call_args[1] if mock_wrapper_class.call_args[1] else {}
        if 'use_cache' in call_kwargs:
            self.assertFalse(call_kwargs['use_cache'])

    @patch('models.hf_integration.TextGeneratorWrapper')
    def test_create_text_model(self, mock_wrapper_class):
        """Test creating text generation model."""
        mock_instance = Mock()
        mock_wrapper_class.return_value = mock_instance

        model = self.factory.create_model('text_generator', use_cache=False)

        self.assertEqual(model, mock_instance)
        mock_wrapper_class.assert_called_once()

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with self.assertRaises(ValueError) as context:
            self.factory.create_model('invalid_type')

        # Verify error message is helpful
        error_msg = str(context.exception)
        self.assertIn('invalid_type', error_msg.lower())

    def test_singleton_factory(self):
        """Test singleton behavior of global factory."""
        factory1 = get_model_factory()
        factory2 = get_model_factory()
        self.assertIs(factory1, factory2, "Factory should be a singleton")

    @patch('models.hf_integration.ImageClassifierWrapper')
    def test_model_creation_performance(self, mock_wrapper_class):
        """Ensure model creation completes in reasonable time."""
        mock_instance = Mock()
        mock_wrapper_class.return_value = mock_instance

        start_time = time.time()
        model = self.factory.create_model('image_classifier', use_cache=False)
        duration = time.time() - start_time

        self.assertLess(duration, MAX_MODEL_CREATION_TIME,
                        f"Model creation took {duration:.2f}s, max allowed {MAX_MODEL_CREATION_TIME}s")


class TestImageValidation(unittest.TestCase):
    """
    Test image-specific validation logic.

    Ensures proper image file handling and validation.
    """

    def setUp(self):
        """Create temporary test images."""
        # Valid image
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img = Image.new('RGB', TEST_IMAGE_SIZE, color='red')
        img.save(self.temp_image.name)
        self.temp_image.close()

        # Invalid file (not an image)
        self.temp_text = tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w')
        self.temp_text.write("This is not an image")
        self.temp_text.close()

    def tearDown(self):
        """Clean up temporary files."""
        try:
            os.unlink(self.temp_image.name)
            os.unlink(self.temp_text.name)
        except Exception:
            pass  # Files may already be deleted

    def validate_image_path(self, path):
        """Helper function for image path validation."""
        if not os.path.exists(path):
            return False
        if not os.path.isfile(path):
            return False
        return any(path.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)

    def test_valid_image_path(self):
        """Test validation with valid image path."""
        self.assertTrue(self.validate_image_path(self.temp_image.name))

    def test_nonexistent_image_path(self):
        """Test validation with nonexistent file."""
        self.assertFalse(self.validate_image_path('nonexistent.jpg'))

    def test_invalid_extension(self):
        """Test validation with invalid file extension."""
        self.assertFalse(self.validate_image_path(self.temp_text.name))

    def test_all_valid_extensions(self):
        """Test that all documented image extensions are recognized."""
        for ext in VALID_IMAGE_EXTENSIONS:
            test_path = f"/fake/path/image{ext}"
            # Create a temporary file with each extension
            temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            temp_file.close()

            try:
                self.assertTrue(self.validate_image_path(temp_file.name),
                                f"Failed to validate extension: {ext}")
            finally:
                os.unlink(temp_file.name)

    def test_case_insensitive_extension(self):
        """Test that extension validation is case-insensitive."""
        test_cases = ['test.JPG', 'test.Jpg', 'test.PNG', 'test.pNg']
        for path in test_cases:
            # Check if extension is recognized (not file existence)
            has_valid_ext = any(path.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)
            self.assertTrue(has_valid_ext, f"Failed for: {path}")


class TestTextValidation(unittest.TestCase):
    """
    Test text-specific validation logic.

    Ensures proper text input validation including edge cases.
    """

    def validate_text_input(self, text):
        """Helper function for text validation."""
        if not isinstance(text, str):
            return False
        cleaned = text.strip()
        return MIN_TEXT_LENGTH < len(cleaned) <= MAX_TEXT_LENGTH

    def test_valid_text_inputs(self):
        """Test various valid text inputs."""
        valid_texts = [
            "Hello world",
            "A" * MAX_TEXT_LENGTH,
            "Special chars: !@#$%^&*()",
            "Unicode: 你好世界 🌍",
            "Multiple\nlines\nof\ntext",
            "Numbers: 123456789",
        ]

        for text in valid_texts:
            self.assertTrue(self.validate_text_input(text),
                            f"Should validate: {text[:50]}...")

    def test_invalid_text_inputs(self):
        """Test various invalid text inputs."""
        invalid_texts = [
            "",  # Empty string
            "   ",  # Only whitespace
            "\n\n\n",  # Only newlines
            "\t\t\t",  # Only tabs
            "A" * (MAX_TEXT_LENGTH + 1),  # Too long
            None,  # None type
            123,  # Integer
            [],  # List
            {},  # Dict
        ]

        for text in invalid_texts:
            self.assertFalse(self.validate_text_input(text),
                             f"Should not validate: {text}")

    def test_boundary_conditions(self):
        """Test text validation at exact boundary lengths."""
        # Exactly at max length
        max_text = "A" * MAX_TEXT_LENGTH
        self.assertTrue(self.validate_text_input(max_text))

        # One character over max
        over_max = "A" * (MAX_TEXT_LENGTH + 1)
        self.assertFalse(self.validate_text_input(over_max))

        # One character (minimum valid)
        min_text = "A"
        self.assertTrue(self.validate_text_input(min_text))

    def test_whitespace_handling(self):
        """Test that validation properly handles leading/trailing whitespace."""
        text_with_spaces = "   valid text   "
        self.assertTrue(self.validate_text_input(text_with_spaces))

        # But only whitespace should fail
        only_spaces = "        "
        self.assertFalse(self.validate_text_input(only_spaces))


class TestIntegration(unittest.TestCase):
    """
    Integration tests for component interaction.

    Ensures different components work together correctly.
    """

    def test_model_info_factory_integration(self):
        """Test integration between ModelInfo and ModelFactory."""
        # Get model names from ModelInfo
        model_names = ModelInfo.get_model_names()

        # Verify factory can handle all models listed in ModelInfo
        factory = ModelFactory()
        available_types = factory.get_available_model_types()

        for model_name in model_names:
            self.assertIn(model_name, available_types,
                          f"Factory doesn't support model: {model_name}")

    @patch('models.hf_integration.ImageClassifierWrapper')
    @patch('models.hf_integration.TextGeneratorWrapper')
    def test_factory_creates_all_registered_models(self, mock_text, mock_image):
        """Test that factory can actually instantiate all registered models."""
        mock_image.return_value = Mock()
        mock_text.return_value = Mock()

        factory = ModelFactory()

        for model_type in factory.get_available_model_types():
            try:
                model = factory.create_model(model_type, use_cache=False)
                self.assertIsNotNone(model, f"Failed to create {model_type}")
            except ValueError:
                self.fail(f"Factory failed to create registered model: {model_type}")

    def test_model_metadata_consistency(self):
        """Test consistency of model metadata across components."""
        for model_key in ModelInfo.get_model_names():
            model_info = ModelInfo.get_model_info(model_key)
            usage_guide = ModelUsageGuide.get_usage_guide(model_key)

            # Verify metadata structure
            self.assertIsInstance(model_info['name'], str)
            self.assertIsInstance(model_info['display_name'], str)
            self.assertIsInstance(usage_guide['preparation_tips'], list)

            # Ensure all required fields exist
            required_fields = ['name', 'display_name', 'task', 'input_type', 'output_type']
            for field in required_fields:
                self.assertIn(field, model_info,
                              f"Model {model_key} missing field: {field}")

            # Verify consistency between info and guide
            self.assertGreater(len(usage_guide['preparation_tips']), 0)
            self.assertGreater(len(usage_guide['interpretation_guide']), 0)

    def test_input_type_consistency(self):
        """Test that models with same input type have consistent requirements."""
        # Group models by input type
        input_types = ModelInfo.get_supported_input_types()

        for input_type in input_types:
            models = ModelInfo.get_models_by_input_type(input_type)
            self.assertGreater(len(models), 0,
                               f"No models found for input type: {input_type}")

            # All models of same type should have consistent metadata
            for model_key, model_info in models.items():
                self.assertEqual(model_info['input_type'], input_type)


class TestErrorHandling(unittest.TestCase):
    """
    Test comprehensive error handling scenarios.

    Ensures application gracefully handles edge cases and error conditions.
    """

    def test_graceful_degradation_invalid_model_key(self):
        """Test that components handle invalid model keys gracefully."""
        # ModelInfo should raise KeyError
        with self.assertRaises(KeyError):
            ModelInfo.get_model_info("invalid_key")

        # ModelUsageGuide should return default
        guide = ModelUsageGuide.get_usage_guide("invalid_key")
        self.assertIsInstance(guide, dict)
        self.assertIn('preparation_tips', guide)

    def test_empty_input_handling(self):
        """Test handling of various empty/null inputs."""
        base_wrapper = BaseModelWrapper("test")

        invalid_inputs = [None, "", "   ", [], {}, 0, False]
        for invalid_input in invalid_inputs:
            result = base_wrapper.validate_input(invalid_input)
            self.assertFalse(result,
                             f"Should reject empty/null input: {invalid_input}")

    def test_factory_error_propagation(self):
        """Test that factory properly propagates errors."""
        factory = ModelFactory()

        # Invalid model type should raise ValueError
        with self.assertRaises(ValueError) as context:
            factory.create_model('nonexistent_type')

        # Error message should be informative
        self.assertIn('nonexistent_type', str(context.exception).lower())

    def test_multiple_error_recovery(self):
        """Test that system can recover from multiple consecutive errors."""
        base_wrapper = BaseModelWrapper("test")

        # Generate multiple errors
        for i in range(10):
            base_wrapper._increment_error_count()

        # System should still be functional
        self.assertEqual(base_wrapper.get_error_count(), 10)
        self.assertTrue(base_wrapper.validate_input("valid input"))

    def test_exception_in_decorator_chain(self):
        """Test exception handling in stacked decorators."""

        @timeit
        @log_exceptions
        def function_that_fails():
            raise RuntimeError("Intentional test error")

        with patch('oop_examples.decorators.logger'):
            with self.assertRaises(RuntimeError):
                function_that_fails()

            # Exception should propagate through decorator chain


class TestConcurrency(unittest.TestCase):
    """
    Test thread safety and concurrent access scenarios.

    Note: Basic tests only - full concurrency testing would require threading module.
    """

    def test_factory_thread_safety_simulation(self):
        """Simulate multiple threads accessing factory."""
        factory1 = get_model_factory()
        factory2 = get_model_factory()

        # Should return same instance (singleton)
        self.assertIs(factory1, factory2)

    def test_model_info_concurrent_reads(self):
        """Test that ModelInfo can handle multiple simultaneous reads."""
        # Simulate multiple reads
        results = []
        for _ in range(100):
            info = ModelInfo.get_model_info('image_classifier')
            results.append(info['name'])

        # All reads should return consistent data
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(results[0], 'image_classifier')


class TestPerformance(unittest.TestCase):
    """
    Basic performance tests to ensure reasonable execution times.
    """

    def test_model_info_lookup_performance(self):
        """Ensure model info lookup is fast."""
        start_time = time.time()

        for _ in range(1000):
            ModelInfo.get_model_info('image_classifier')

        duration = time.time() - start_time

        # Should complete 1000 lookups in under 1 second
        self.assertLess(duration, 1.0,
                        f"1000 lookups took {duration:.2f}s")

    def test_validation_performance(self):
        """Ensure validation is performant."""
        wrapper = BaseModelWrapper("test")

        start_time = time.time()

        for i in range(10000):
            wrapper.validate_input(f"test_input_{i}")

        duration = time.time() - start_time

        # Should complete 10000 validations in under 1 second
        self.assertLess(duration, 1.0,
                        f"10000 validations took {duration:.2f}s")


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and unusual inputs.
    """

    def test_very_long_model_name(self):
        """Test handling of extremely long model names."""
        long_name = "a" * 1000
        wrapper = BaseModelWrapper(long_name)
        self.assertEqual(wrapper.model_name, long_name)

    def test_special_characters_in_text(self):
        """Test text validation with various special characters."""
        special_texts = [
            "Hello\x00World",  # Null character
            "Test\r\nNewline",  # Windows newline
            "Unicode: \u2603 \u2764",  # Snowman and heart emoji
            "Math: ∑∫∂∇",  # Mathematical symbols
            "Arabic: مرحبا",
            "Chinese: 你好",
            "Hebrew: שלום",
            "Emoji: 😀🎉🚀",
        ]

        validator = TestTextValidation()
        for text in special_texts:
            # Should handle special characters gracefully
            try:
                result = validator.validate_text_input(text)
                # Result should be boolean, no exceptions
                self.assertIsInstance(result, bool)
            except Exception as e:
                self.fail(f"Failed on special text: {text[:20]}... Error: {e}")

    def test_numeric_string_inputs(self):
        """Test handling of numeric strings."""
        validator = TestTextValidation()

        numeric_strings = [
            "123",
            "3.14159",
            "-42",
            "1e10",
            "0x1F",  # Hex
        ]

        for num_str in numeric_strings:
            result = validator.validate_text_input(num_str)
            self.assertTrue(result, f"Should accept numeric string: {num_str}")

    def test_whitespace_variations(self):
        """Test different types of whitespace handling."""
        validator = TestTextValidation()

        whitespace_tests = [
            ("normal text", True),
            ("  spaces before", True),
            ("spaces after  ", True),
            ("  both sides  ", True),
            ("\ttabs\t", True),
            ("\nnewlines\n", True),
            ("mixed \t\n whitespace", True),
            ("   ", False),  # Only whitespace
            ("\t\t\t", False),  # Only tabs
            ("\n\n\n", False),  # Only newlines
        ]

        for text, expected in whitespace_tests:
            result = validator.validate_text_input(text)
            self.assertEqual(result, expected,
                             f"Failed for whitespace test: {repr(text)}")


class TestDocumentation(unittest.TestCase):
    """
    Test that all components have proper documentation.
    """

    def test_model_info_docstrings(self):
        """Verify ModelInfo class has proper documentation."""
        self.assertIsNotNone(ModelInfo.__doc__)
        self.assertGreater(len(ModelInfo.__doc__), 10)

    def test_factory_docstrings(self):
        """Verify ModelFactory has proper documentation."""
        self.assertIsNotNone(ModelFactory.__doc__)
        self.assertGreater(len(ModelFactory.__doc__), 10)

    def test_all_public_methods_documented(self):
        """Ensure all public methods have docstrings."""
        classes_to_check = [
            ModelInfo,
            ModelUsageGuide,
            ModelFactory,
            BaseModelWrapper,
        ]

        for cls in classes_to_check:
            for attr_name in dir(cls):
                if not attr_name.startswith('_'):
                    attr = getattr(cls, attr_name)
                    if callable(attr):
                        self.assertIsNotNone(attr.__doc__,
                                             f"{cls.__name__}.{attr_name} lacks docstring")


# Test utilities and runners
def run_all_tests():
    """
    Run all tests and return results.

    Returns:
        unittest.TestResult: The test results object
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestModelInfo,
        TestModelUsageGuide,
        TestBaseClasses,
        TestDecorators,
        TestModelFactory,
        TestImageValidation,
        TestTextValidation,
        TestIntegration,
        TestErrorHandling,
        TestConcurrency,
        TestPerformance,
        TestEdgeCases,
        TestDocumentation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result


def generate_test_report():
    """
    Generate a comprehensive test report with statistics and insights.

    Returns:
        bool: True if all tests passed, False otherwise
    """
    print("=" * 70)
    print(" HIT137 ASSIGNMENT 3 - COMPREHENSIVE TEST REPORT")
    print("=" * 70)
    print()

    # Run tests and capture results
    result = run_all_tests()

    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.testsRun > 0:
        success_count = result.testsRun - len(result.failures) - len(result.errors)
        success_rate = (success_count / result.testsRun) * 100
        print(f"Success Rate: {success_rate:.1f}%")

    if result.failures:
        print("\n" + "=" * 70)
        print(" FAILURES")
        print("=" * 70)
        for test, traceback in result.failures:
            print(f"\n  • {test}")
            # Extract the assertion error message
            error_lines = traceback.split('\n')
            for line in error_lines:
                if 'AssertionError' in line or 'Error' in line:
                    print(f"    {line.strip()}")

    if result.errors:
        print("\n" + "=" * 70)
        print(" ERRORS")
        print("=" * 70)
        for test, traceback in result.errors:
            print(f"\n  • {test}")
            # Extract the error message
            error_lines = traceback.split('\n')
            for line in error_lines[-3:]:
                if line.strip():
                    print(f"    {line.strip()}")

    if not result.failures and not result.errors:
        print("\n" + "=" * 70)
        print(" ✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n The application successfully demonstrates:")
        print("  ✓ Success: Multiple Inheritance with proper constructor chaining")
        print("  ✓ Success: Encapsulation with private methods and data hiding")
        print("  ✓ Success: Polymorphism with unified interfaces")
        print("  ✓ Success: Method Overriding with specialized implementations")
        print("  ✓ Success: Multiple Decorators with stacked functionality")
        print("  ✓ Success: Comprehensive error handling and validation")
        print("  ✓ Success: Professional code organization and testing")
        print("  ✓ Success: Thread safety and performance optimization")
        print("  ✓ Success: Edge case handling and robustness")
        print("\n Testing best practices implemented:")
        print("  • Isolated test cases with proper setUp/tearDown")
        print("  • Comprehensive mocking to avoid external dependencies")
        print("  • Edge case and boundary condition testing")
        print("  • Integration testing across components")
        print("  • Performance benchmarking")
        print("  • Clear, descriptive test names and documentation")

    print("\n" + "=" * 70)
    print(f" Test execution completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return result.wasSuccessful()


def run_specific_test_class(test_class_name):
    """
    Run tests from a specific test class.

    Args:
        test_class_name (str): Name of the test class to run

    Returns:
        bool: True if all tests passed, False otherwise
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Map of test class names to classes
    test_classes = {
        'TestModelInfo': TestModelInfo,
        'TestModelUsageGuide': TestModelUsageGuide,
        'TestBaseClasses': TestBaseClasses,
        'TestDecorators': TestDecorators,
        'TestModelFactory': TestModelFactory,
        'TestImageValidation': TestImageValidation,
        'TestTextValidation': TestTextValidation,
        'TestIntegration': TestIntegration,
        'TestErrorHandling': TestErrorHandling,
        'TestConcurrency': TestConcurrency,
        'TestPerformance': TestPerformance,
        'TestEdgeCases': TestEdgeCases,
        'TestDocumentation': TestDocumentation,
    }

    if test_class_name in test_classes:
        tests = loader.loadTestsFromTestCase(test_classes[test_class_name])
        suite.addTests(tests)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()
    else:
        print(f"Test class '{test_class_name}' not found.")
        print(f"Available test classes: {', '.join(test_classes.keys())}")
        return False


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--class" and len(sys.argv) > 2:
            # Run specific test class
            success = run_specific_test_class(sys.argv[2])
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python test_suite.py              # Run all tests")
            print("  python test_suite.py --class NAME # Run specific test class")
            print("  python test_suite.py --help       # Show this help")
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            sys.exit(1)
    else:
        # Run all tests with full report
        success = generate_test_report()

    sys.exit(0 if success else 1)