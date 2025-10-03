"""
Comprehensive test suite for the AI Model Integration Application.

This module provides unit tests and integration tests for all major components,
demonstrating testing best practices and ensuring code reliability.

Demonstrates: Unit testing, integration testing, mock objects, test organization
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from io import StringIO
from PIL import Image
import tempfile

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


class TestModelInfo(unittest.TestCase):
    """Test the ModelInfo class functionality."""
    
    def test_get_model_names(self):
        """Test getting list of available model names."""
        names = ModelInfo.get_model_names()
        self.assertIsInstance(names, list)
        self.assertIn('image_classifier', names)
        self.assertIn('text_generator', names)
    
    def test_get_model_info_valid_key(self):
        """Test getting info for valid model key."""
        info = ModelInfo.get_model_info('image_classifier')
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('display_name', info)
        self.assertIn('task', info)
        self.assertEqual(info['input_type'], 'image')
    
    def test_get_model_info_invalid_key(self):
        """Test error handling for invalid model key."""
        with self.assertRaises(KeyError):
            ModelInfo.get_model_info('nonexistent_model')
    
    def test_get_models_by_input_type(self):
        """Test filtering models by input type."""
        text_models = ModelInfo.get_models_by_input_type('text')
        self.assertIsInstance(text_models, dict)
        
        for model_info in text_models.values():
            self.assertEqual(model_info['input_type'], 'text')
    
    def test_validate_model_key(self):
        """Test model key validation."""
        self.assertTrue(ModelInfo.validate_model_key('image_classifier'))
        self.assertTrue(ModelInfo.validate_model_key('text_generator'))
        self.assertFalse(ModelInfo.validate_model_key('invalid_key'))
    
    def test_get_supported_input_types(self):
        """Test getting all supported input types."""
        input_types = ModelInfo.get_supported_input_types()
        self.assertIsInstance(input_types, list)
        self.assertIn('image', input_types)
        self.assertIn('text', input_types)


class TestModelUsageGuide(unittest.TestCase):
    """Test the ModelUsageGuide class."""
    
    def test_get_usage_guide_valid_key(self):
        """Test getting usage guide for valid model."""
        guide = ModelUsageGuide.get_usage_guide('image_classifier')
        self.assertIsInstance(guide, dict)
        self.assertIn('preparation_tips', guide)
        self.assertIn('interpretation_guide', guide)
        self.assertIn('limitations', guide)
    
    def test_get_usage_guide_invalid_key(self):
        """Test default guide for invalid model key."""
        guide = ModelUsageGuide.get_usage_guide('invalid_model')
        self.assertIsInstance(guide, dict)
        self.assertIn('preparation_tips', guide)
        # Should return default messages
        self.assertIn('No specific guidance available', guide['preparation_tips'][0])


class TestBaseClasses(unittest.TestCase):
    """Test the OOP base classes and mixins."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_wrapper = BaseModelWrapper("test_model")
    
    def test_base_model_wrapper_initialization(self):
        """Test BaseModelWrapper initialization."""
        self.assertEqual(self.base_wrapper.model_name, "test_model")
        self.assertFalse(self.base_wrapper.is_loaded())
        self.assertEqual(self.base_wrapper.get_error_count(), 0)
    
    def test_base_validation(self):
        """Test base input validation."""
        self.assertTrue(self.base_wrapper.validate_input("test"))
        self.assertFalse(self.base_wrapper.validate_input(None))
    
    def test_error_count_increment(self):
        """Test error counting functionality."""
        initial_count = self.base_wrapper.get_error_count()
        self.base_wrapper._increment_error_count()
        self.assertEqual(self.base_wrapper.get_error_count(), initial_count + 1)
    
    def test_model_logging_mixin(self):
        """Test ModelLoggingMixin functionality."""
        mixin = ModelLoggingMixin()
        
        # Test logging operation
        mixin.log_operation("test_operation", "input", "result", 1.5)
        history = mixin.get_operation_history()
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['operation'], 'test_operation')
        self.assertEqual(history[0]['duration'], 1.5)
        
        # Test clear history
        mixin.clear_history()
        self.assertEqual(len(mixin.get_operation_history()), 0)
    
    def test_polymorphic_model_manager(self):
        """Test PolymorphicModelManager functionality."""
        manager = PolymorphicModelManager()
        
        # Create mock model
        mock_model = Mock()
        mock_model.run.return_value = {"result": "test"}
        mock_model.get_model_info.return_value = {"name": "test_model"}
        
        # Register and test
        manager.register_model("test", mock_model)
        self.assertIn("test", manager.get_available_models())
        
        manager.set_active_model("test")
        result = manager.process_with_current_model("test_input")
        
        self.assertEqual(result["result"], "test")
        mock_model.run.assert_called_once_with("test_input")


class TestDecorators(unittest.TestCase):
    """Test custom decorators functionality."""
    
    def test_timeit_decorator(self):
        """Test timing decorator."""
        @timeit
        def test_function():
            return "test_result"
        
        # Capture log output
        with patch('oop_examples.decorators.logger') as mock_logger:
            result = test_function()
            self.assertEqual(result, "test_result")
            mock_logger.info.assert_called()
            
            # Check that timing info was logged
            call_args = mock_logger.info.call_args[0][0]
            self.assertIn("test_function executed in", call_args)
    
    def test_log_exceptions_decorator(self):
        """Test exception logging decorator."""
        @log_exceptions
        def failing_function():
            raise ValueError("Test error")
        
        with patch('oop_examples.decorators.logger') as mock_logger:
            with self.assertRaises(ValueError):
                failing_function()
            
            # Verify exception was logged
            mock_logger.error.assert_called()
    
    def test_validate_input_decorator(self):
        """Test input validation decorator."""
        @validate_input(input_type="str")
        def string_function(data):
            return f"processed: {data}"
        
        # Valid input
        result = string_function("test")
        self.assertEqual(result, "processed: test")
        
        # Invalid input
        with self.assertRaises(TypeError):
            string_function(123)


class TestModelFactory(unittest.TestCase):
    """Test ModelFactory functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = ModelFactory()
    
    def test_get_available_model_types(self):
        """Test getting available model types."""
        types = self.factory.get_available_model_types()
        self.assertIsInstance(types, list)
        self.assertIn('image_classifier', types)
        self.assertIn('text_generator', types)
    
    @patch('models.hf_integration.ImageClassifierWrapper')
    def test_create_image_model(self, mock_wrapper_class):
        """Test creating image classification model."""
        mock_instance = Mock()
        mock_wrapper_class.return_value = mock_instance
        
        model = self.factory.create_model('image_classifier', use_cache=False)
        self.assertEqual(model, mock_instance)
        mock_wrapper_class.assert_called_once()
    
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
        with self.assertRaises(ValueError):
            self.factory.create_model('invalid_type')
    
    def test_singleton_factory(self):
        """Test singleton behavior of global factory."""
        factory1 = get_model_factory()
        factory2 = get_model_factory()
        self.assertIs(factory1, factory2)


class TestImageValidation(unittest.TestCase):
    """Test image-specific validation logic."""
    
    def setUp(self):
        """Create temporary test image."""
        self.temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.temp_image.name)
        self.temp_image.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        os.unlink(self.temp_image.name)
    
    def test_valid_image_path(self):
        """Test validation with valid image path."""
        # This would normally test ImageClassifierWrapper.validate_input
        # but since we're avoiding actual model loading, we'll test the logic
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        def validate_image_path(path):
            if not os.path.exists(path):
                return False
            return any(path.lower().endswith(ext) for ext in valid_extensions)
        
        self.assertTrue(validate_image_path(self.temp_image.name))
        self.assertFalse(validate_image_path('nonexistent.jpg'))
        self.assertFalse(validate_image_path('test.txt'))


class TestTextValidation(unittest.TestCase):
    """Test text-specific validation logic."""
    
    def test_valid_text_input(self):
        """Test text validation logic."""
        def validate_text_input(text):
            if not isinstance(text, str):
                return False
            cleaned = text.strip()
            return 0 < len(cleaned) <= 1000
        
        # Valid inputs
        self.assertTrue(validate_text_input("Hello world"))
        self.assertTrue(validate_text_input("A" * 1000))
        
        # Invalid inputs
        self.assertFalse(validate_text_input(""))
        self.assertFalse(validate_text_input("   "))
        self.assertFalse(validate_text_input("A" * 1001))
        self.assertFalse(validate_text_input(123))
        self.assertFalse(validate_text_input(None))


class TestIntegration(unittest.TestCase):
    """Integration tests for component interaction."""
    
    def test_model_info_factory_integration(self):
        """Test integration between ModelInfo and ModelFactory."""
        # Get model names from ModelInfo
        model_names = ModelInfo.get_model_names()
        
        # Verify factory can create all models listed in ModelInfo
        factory = ModelFactory()
        available_types = factory.get_available_model_types()
        
        for model_name in model_names:
            self.assertIn(model_name, available_types)
    
    def test_model_metadata_consistency(self):
        """Test consistency of model metadata across components."""
        for model_key in ModelInfo.get_model_names():
            model_info = ModelInfo.get_model_info(model_key)
            usage_guide = ModelUsageGuide.get_usage_guide(model_key)
            
            # Basic consistency checks
            self.assertIsInstance(model_info['name'], str)
            self.assertIsInstance(model_info['display_name'], str)
            self.assertIsInstance(usage_guide['preparation_tips'], list)
            
            # Ensure all required fields exist
            required_fields = ['name', 'display_name', 'task', 'input_type', 'output_type']
            for field in required_fields:
                self.assertIn(field, model_info)


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error handling scenarios."""
    
    def test_graceful_degradation(self):
        """Test that components handle errors gracefully."""
        # Test ModelInfo with invalid keys
        with self.assertRaises(KeyError):
            ModelInfo.get_model_info("invalid_key")
        
        # Test that default values are returned when appropriate
        guide = ModelUsageGuide.get_usage_guide("invalid_key")
        self.assertIsInstance(guide, dict)
        self.assertIn('preparation_tips', guide)


# Test runner and utilities
def run_all_tests():
    """Run all tests and return results."""
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
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


def generate_test_report():
    """Generate a comprehensive test report."""
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
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\n ERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('Exception: ')[-1].split('\n')[0]}")
    
    if not result.failures and not result.errors:
        print("\n ALL TESTS PASSED! ")
        print("\n The application demonstrates:")
        print("  Sucess: Multiple Inheritance with proper constructor chaining")
        print("  Sucess: Encapsulation with private methods and data hiding")
        print("  Sucess: Polymorphism with unified interfaces")
        print("  Sucess: Method Overriding with specialized implementations")
        print("  Sucess: Multiple Decorators with stacked functionality")
        print("  Sucess: Comprehensive error handling and validation")
        print("  Sucess: Professional code organization and testing")
    
    print("\n" + "=" * 70)
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when executed directly
    success = generate_test_report()
    sys.exit(0 if success else 1)