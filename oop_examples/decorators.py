"""
Custom decorators for the AI Model Integration Application.

This module implements reusable decorators that demonstrate advanced Python concepts
and provide functionality like timing, logging, and error handling.

Demonstrates: Multiple decorators, decorator stacking, functional programming
"""

import time
import functools
import logging
from typing import Any, Callable, Dict


# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of functions.
    
    This decorator demonstrates:
    - Decorator pattern implementation
    - Performance monitoring
    - Function wrapping with functools
    
    Args:
        func: The function to be timed
        
    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Execute the original function
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log the execution time
        logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper


def log_exceptions(func: Callable) -> Callable:
    """
    Decorator to catch and log exceptions while allowing them to propagate.
    
    This decorator demonstrates:
    - Exception handling patterns
    - Logging best practices
    - Decorator chaining compatibility
    
    Args:
        func: The function to wrap with exception logging
        
    Returns:
        Wrapped function that logs exceptions
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception with context
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(f"Arguments: args={args}, kwargs={kwargs}")
            
            # Re-raise the exception to maintain normal error flow
            raise
    
    return wrapper


def validate_input(input_type: str = None, required_keys: list = None):
    """
    Parametric decorator for input validation.
    
    This decorator demonstrates:
    - Parametric decorators (decorators that accept arguments)
    - Input validation patterns
    - Flexible validation logic
    
    Args:
        input_type: Expected type of the first argument
        required_keys: Required keys if first argument is a dictionary
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate input if criteria are specified
            if args and input_type:
                if not isinstance(args[0], eval(input_type)):
                    raise TypeError(f"Expected {input_type}, got {type(args[0])}")
            
            if args and required_keys and isinstance(args[0], dict):
                missing_keys = [key for key in required_keys if key not in args[0]]
                if missing_keys:
                    raise ValueError(f"Missing required keys: {missing_keys}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def cache_results(max_size: int = 128):
    """
    Simple caching decorator for model results.
    
    This decorator demonstrates:
    - Caching patterns
    - Memory management
    - Performance optimization
    
    Args:
        max_size: Maximum number of cached results
        
    Returns:
        Decorator that caches function results
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Any] = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            # Return cached result if available
            if cache_key in cache:
                logger.info(f"Cache hit for {func.__name__}")
                return cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            # Manage cache size
            if len(cache) >= max_size:
                # Remove oldest entry (simple FIFO strategy)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[cache_key] = result
            logger.info(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator to retry function calls on failure.
    
    This decorator demonstrates:
    - Retry patterns
    - Error recovery strategies
    - Configurable behavior
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between retry attempts in seconds
        
    Returns:
        Decorator that retries failed function calls
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    
                    if attempt < max_attempts - 1:  # Don't sleep after the last attempt
                        time.sleep(delay)
            
            # If all attempts failed, raise the last exception
            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        return wrapper
    return decorator


# Example of decorator usage (for demonstration)
if __name__ == "__main__":
    # Example function with multiple decorators
    @timeit
    @log_exceptions
    @cache_results(max_size=10)
    def example_function(data: str) -> str:
        """Example function demonstrating multiple decorator usage."""
        time.sleep(0.1)  # Simulate some processing time
        return f"Processed: {data}"
    
    # Test the decorated function
    print(example_function("test data"))
    print(example_function("test data"))  # Should use cache