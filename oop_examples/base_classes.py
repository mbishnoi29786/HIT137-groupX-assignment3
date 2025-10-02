"""
Base classes and interfaces for the AI Model Integration Application.

This module demonstrates advanced OOP concepts including multiple inheritance,
abstract base classes, mixins, and polymorphism.

Demonstrates: Inheritance, polymorphism, encapsulation, abstract classes
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime


class BaseModelInterface(ABC):
    """
    Abstract base class defining the interface for AI model wrappers.
    
    This demonstrates:
    - Abstract base classes
    - Interface definition
    - Contract enforcement through inheritance
    """
    
    @abstractmethod
    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input data through the AI model.
        
        Args:
            input_data: Input to be processed by the model
            
        Returns:
            Dictionary containing model results
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model metadata
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate if input is appropriate for this model.
        
        Args:
            input_data: Data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass


class ModelLoggingMixin:
    """
    Mixin class providing logging functionality for model operations.
    
    This demonstrates:
    - Mixin pattern
    - Multiple inheritance support
    - Reusable functionality across different model types
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.operation_history: List[Dict[str, Any]] = []
    
    def log_operation(self, operation: str, input_data: Any = None, 
                     result: Any = None, duration: float = None) -> None:
        """
        Log model operations for debugging and monitoring.
        
        Args:
            operation: Description of the operation
            input_data: Input that was processed
            result: Result of the operation
            duration: Time taken for the operation
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'input_type': type(input_data).__name__ if input_data else None,
            'result_type': type(result).__name__ if result else None,
            'duration': duration,
            'success': result is not None
        }
        
        self.operation_history.append(log_entry)
        self.logger.info(f"Operation: {operation}, Duration: {duration}s")
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of operations performed by this model.
        
        Returns:
            List of operation log entries
        """
        return self.operation_history.copy()
    
    def clear_history(self) -> None:
        """Clear the operation history."""
        self.operation_history.clear()


class ModelMetadataMixin:
    """
    Mixin class for managing model metadata and information.
    
    This demonstrates:
    - Information encapsulation
    - Metadata management patterns
    - Reusable model information handling
    """
    
    def __init__(self):
        self._metadata: Dict[str, Any] = {}
        self._initialization_time = datetime.now()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata for the model.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value by key.
        
        Args:
            key: Metadata key to retrieve
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)
    
    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get all metadata as a dictionary.
        
        Returns:
            Complete metadata dictionary
        """
        return {
            **self._metadata,
            'initialization_time': self._initialization_time.isoformat(),
            'uptime_seconds': (datetime.now() - self._initialization_time).total_seconds()
        }


class BaseModelWrapper(BaseModelInterface):
    """
    Base implementation of the model interface with common functionality.
    
    This demonstrates:
    - Concrete base class implementation
    - Template method pattern
    - Common functionality consolidation
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._is_loaded = False
        self._error_count = 0
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._is_loaded
    
    def get_error_count(self) -> int:
        """Get the number of errors encountered."""
        return self._error_count
    
    def _increment_error_count(self) -> None:
        """Increment the error counter (protected method)."""
        self._error_count += 1
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Base validation - can be overridden by subclasses.
        
        Args:
            input_data: Data to validate
            
        Returns:
            True if input is not None
        """
        return input_data is not None
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get basic model information.
        
        Returns:
            Dictionary with basic model info
        """
        return {
            'name': self.model_name,
            'class': self.__class__.__name__,
            'loaded': str(self._is_loaded),
            'errors': str(self._error_count)
        }


class MultiInheritanceModelWrapper(BaseModelWrapper, ModelLoggingMixin, ModelMetadataMixin):
    """
    Example of multiple inheritance combining base functionality with mixins.
    
    This demonstrates:
    - Multiple inheritance
    - Mixin composition
    - Method resolution order (MRO)
    - Diamond problem resolution
    """
    
    def __init__(self, model_name: str):
        # Call all parent constructors properly
        BaseModelWrapper.__init__(self, model_name)
        ModelLoggingMixin.__init__(self)
        ModelMetadataMixin.__init__(self)
        
        # Set initial metadata
        self.set_metadata('model_name', model_name)
        self.set_metadata('wrapper_type', 'MultiInheritance')
    
    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Template implementation of the run method.
        
        Args:
            input_data: Input to process
            
        Returns:
            Processing results
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        # Log the operation start
        self.log_operation("run_started", input_data)
        
        try:
            # This would be implemented by concrete subclasses
            result = self._process_input(input_data)
            
            # Log successful operation
            self.log_operation("run_completed", input_data, result)
            
            return result
        
        except Exception as e:
            self._increment_error_count()
            self.log_operation("run_failed", input_data)
            raise
    
    def _process_input(self, input_data: Any) -> Dict[str, Any]:
        """
        Template method to be implemented by subclasses.
        
        This demonstrates the Template Method pattern.
        """
        raise NotImplementedError("Subclasses must implement _process_input")
    
    def get_comprehensive_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information combining all aspects.
        
        Returns:
            Complete information dictionary
        """
        info = self.get_model_info()
        info.update(self.get_all_metadata())
        info['operation_count'] = len(self.operation_history)
        
        return info


class PolymorphicModelManager:
    """
    Manager class demonstrating polymorphism with different model types.
    
    This demonstrates:
    - Polymorphism in action
    - Strategy pattern
    - Dynamic model switching
    """
    
    def __init__(self):
        self._models: Dict[str, BaseModelInterface] = {}
        self._current_model: Optional[str] = None
    
    def register_model(self, name: str, model: BaseModelInterface) -> None:
        """
        Register a model with the manager.
        
        Args:
            name: Unique name for the model
            model: Model instance implementing BaseModelInterface
        """
        self._models[name] = model
    
    def set_active_model(self, name: str) -> None:
        """
        Set the active model by name.
        
        Args:
            name: Name of the model to activate
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        self._current_model = name
    
    def process_with_current_model(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input with the currently active model.
        
        This demonstrates polymorphism - the same interface call
        works with any model type.
        
        Args:
            input_data: Data to process
            
        Returns:
            Processing results
        """
        if not self._current_model:
            raise ValueError("No active model set")
        
        model = self._models[self._current_model]
        return model.run(input_data)
    
    def get_available_models(self) -> List[str]:
        """
        Get list of registered model names.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, str]:
        """
        Get information about a specific model.
        
        Args:
            name: Model name
            
        Returns:
            Model information
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")
        
        return self._models[name].get_model_info()


# Example usage and testing
if __name__ == "__main__":
    # This demonstrates the class hierarchy and polymorphism
    
    class ExampleModel(MultiInheritanceModelWrapper):
        """Example concrete model implementation."""
        
        def _process_input(self, input_data: Any) -> Dict[str, Any]:
            return {
                'input': str(input_data),
                'output': f"Processed: {input_data}",
                'model': self.model_name
            }
    
    # Create and test the model
    model = ExampleModel("example-model")
    result = model.run("test input")
    print(f"Result: {result}")
    print(f"Model info: {model.get_comprehensive_info()}")
    
    # Demonstrate polymorphism with manager
    manager = PolymorphicModelManager()
    manager.register_model("example", model)
    manager.set_active_model("example")
    
    poly_result = manager.process_with_current_model("polymorphic test")
    print(f"Polymorphic result: {poly_result}")