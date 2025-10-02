"""
Model factory for creating and managing AI model instances.

This module provides a centralized factory for creating model wrappers
and managing model lifecycle. Demonstrates Factory pattern and model management.

Demonstrates: Factory pattern, Singleton pattern, model lifecycle management
"""

from typing import Dict, Optional, Type, Any
import logging
from models.hf_integration import (
    ImageClassifierWrapper,
    SentimentClassifierWrapper,
    SpeechToTextWrapper,
    TextToImageWrapper,
    MultiInheritanceModelWrapper
)
# Updated to use new configuration structure


class ModelFactory:
    """
    Factory class for creating and managing AI model instances.
    
    This demonstrates:
    - Factory pattern implementation
    - Model lifecycle management
    - Type safety and validation
    - Singleton-like behavior for model reuse
    """
    
    # Registry of available model types and their corresponding classes
    _MODEL_REGISTRY: Dict[str, Type[MultiInheritanceModelWrapper]] = {
        'sentiment_classifier': SentimentClassifierWrapper,
        'image_classifier': ImageClassifierWrapper,
        'speech_to_text': SpeechToTextWrapper,
        'text_to_image': TextToImageWrapper
    }
    
    # Cache for model instances (prevents duplicate loading)
    _model_cache: Dict[str, MultiInheritanceModelWrapper] = {}
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._created_models: Dict[str, MultiInheritanceModelWrapper] = {}
    
    @classmethod
    def register_model_type(cls, model_type: str, 
                           wrapper_class: Type[MultiInheritanceModelWrapper]) -> None:
        """
        Register a new model type with the factory.
        
        This allows extending the factory with new model types.
        
        Args:
            model_type: Unique identifier for the model type
            wrapper_class: Class that wraps the model
        """
        if not issubclass(wrapper_class, MultiInheritanceModelWrapper):
            raise TypeError("Wrapper class must inherit from MultiInheritanceModelWrapper")
        
        cls._MODEL_REGISTRY[model_type] = wrapper_class
    
    @classmethod
    def get_available_model_types(cls) -> list[str]:
        """
        Get list of all registered model types.
        
        Returns:
            List of available model type identifiers
        """
        return list(cls._MODEL_REGISTRY.keys())
    
    def create_model(self, model_type: str, model_name: str = None, 
                    use_cache: bool = True) -> MultiInheritanceModelWrapper:
        """
        Create a model wrapper instance.
        
        This demonstrates the Factory pattern - creating objects without
        specifying their exact classes.
        
        Args:
            model_type: Type of model to create ('sentiment_classifier', 'image_classifier', 'speech_to_text', 'text_to_image')
            model_name: Specific model name (uses default if None)
            use_cache: Whether to use cached instances
            
        Returns:
            Model wrapper instance
            
        Raises:
            ValueError: If model_type is not registered
            RuntimeError: If model creation fails
        """
        if model_type not in self._MODEL_REGISTRY:
            available_types = list(self._MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model type '{model_type}'. "
                           f"Available types: {available_types}")
        
        # Generate cache key
        from models.model_info import MODELS_CONFIG
        default_name = MODELS_CONFIG.get(model_type, {}).get('model_name', 'unknown')
        actual_model_name = model_name or default_name
        cache_key = f"{model_type}:{actual_model_name}"
        
        # Check cache first if enabled
        if use_cache and cache_key in self._model_cache:
            self.logger.info(f"Returning cached model: {cache_key}")
            return self._model_cache[cache_key]
        
        try:
            # Create new model instance
            wrapper_class = self._MODEL_REGISTRY[model_type]
            model_instance = wrapper_class(actual_model_name)
            
            # Cache the instance if requested
            if use_cache:
                self._model_cache[cache_key] = model_instance
            
            # Track in instance registry
            self._created_models[cache_key] = model_instance
            
            self.logger.info(f"Created new model: {cache_key}")
            return model_instance
            
        except Exception as e:
            self.logger.error(f"Failed to create model {cache_key}: {e}")
            raise RuntimeError(f"Model creation failed: {e}")
    
    def get_model_by_input_type(self, input_type: str) -> Optional[MultiInheritanceModelWrapper]:
        """
        Get a model that can handle the specified input type.
        
        Args:
            input_type: Type of input ('image', 'text', 'audio')
            
        Returns:
            Model wrapper that can handle the input type, or None
        """
        from models.model_info import MODELS_CONFIG
        
        # Find models that match the input type  
        models_by_input = {}
        for model_key, config in MODELS_CONFIG.items():
            if config['input_type'] == input_type:
                models_by_input[model_key] = config
        
        if not models_by_input:
            return None
        
        # Get the first available model for this input type
        model_key = list(models_by_input.keys())[0]
        return self.create_model(model_key)
    
    def create_all_default_models(self) -> Dict[str, MultiInheritanceModelWrapper]:
        """
        Create instances of all available model types with default configurations.
        
        Returns:
            Dictionary mapping model types to their instances
        """
        models = {}
        
        for model_type in self._MODEL_REGISTRY.keys():
            try:
                model_instance = self.create_model(model_type)
                models[model_type] = model_instance
                self.logger.info(f"Successfully created default {model_type}")
            except Exception as e:
                self.logger.error(f"Failed to create default {model_type}: {e}")
                # Continue with other models even if one fails
        
        return models
    
    def get_created_models(self) -> Dict[str, MultiInheritanceModelWrapper]:
        """
        Get all models created by this factory instance.
        
        Returns:
            Dictionary of created models
        """
        return self._created_models.copy()
    
    def clear_cache(self, model_type: str = None) -> None:
        """
        Clear model cache.
        
        Args:
            model_type: Specific model type to clear, or None to clear all
        """
        if model_type:
            # Clear specific model type from cache
            keys_to_remove = [key for key in self._model_cache.keys() 
                            if key.startswith(f"{model_type}:")]
            for key in keys_to_remove:
                del self._model_cache[key]
                self.logger.info(f"Cleared cached model: {key}")
        else:
            # Clear entire cache
            self._model_cache.clear()
            self.logger.info("Cleared entire model cache")
    
    def get_model_info_summary(self) -> Dict[str, Any]:
        """
        Get summary information about all available models.
        
        Returns:
            Dictionary with model information summary
        """
        summary = {
            'available_types': self.get_available_model_types(),
            'cached_models': list(self._model_cache.keys()),
            'created_models': list(self._created_models.keys()),
            'model_details': {}
        }
        
        # Add detailed info for each model type
        from models.model_info import MODELS_CONFIG
        for model_type in self._MODEL_REGISTRY.keys():
            try:
                model_info = MODELS_CONFIG.get(model_type, {})
                summary['model_details'][model_type] = {
                    'name': model_info.get('model_name', 'Unknown'),
                    'display_name': model_info.get('display_name', 'Unknown'),
                    'task': model_info.get('task', 'unknown'),
                    'input_type': model_info.get('input_type', 'unknown'),
                    'description': model_info['description'][:100] + "..."  # Truncate for summary
                }
            except Exception as e:
                summary['model_details'][model_type] = {'error': str(e)}
        
        return summary


# Singleton instance for global model factory access
_global_factory = None

def get_model_factory() -> ModelFactory:
    """
    Get the global model factory instance (Singleton pattern).
    
    This demonstrates the Singleton pattern - ensuring only one
    factory instance exists globally.
    
    Returns:
        Global ModelFactory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = ModelFactory()
    return _global_factory


def create_model(model_type: str, model_name: str = None) -> MultiInheritanceModelWrapper:
    """
    Convenience function to create a model using the global factory.
    
    Args:
        model_type: Type of model to create
        model_name: Specific model name (optional)
        
    Returns:
        Model wrapper instance
    """
    factory = get_model_factory()
    return factory.create_model(model_type, model_name)


# Example usage and testing
if __name__ == "__main__":
    # Test the factory
    factory = ModelFactory()
    
    print("Available model types:", factory.get_available_model_types())
    
    try:
        # Test creating different model types
        sentiment_model = factory.create_model('sentiment_classifier')
        img_model = factory.create_model('image_classifier')
        
        print(f"Created sentiment model: {sentiment_model.model_name}")
        print(f"Created image model: {img_model.model_name}")
        
        # Test cache functionality
        cached_model = factory.create_model('sentiment_classifier')  # Should use cache
        print(f"Cache test - same instance: {cached_model is sentiment_model}")
        
        # Test model info summary
        summary = factory.get_model_info_summary()
        print(f"Model summary: {len(summary['available_types'])} types available")
        
    except Exception as e:
        print(f"Factory test error: {e}")