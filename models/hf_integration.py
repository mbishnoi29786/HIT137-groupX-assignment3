"""
Hugging Face model integration wrappers for the AI Model Integration Application.

This module provides wrapper classes for Hugging Face models that demonstrate
advanced OOP concepts while providing a clean interface for model usage.

Demonstrates: Multiple inheritance, encapsulation, polymorphism, method overriding
"""

import os
import tempfile
from typing import Dict, Any, List, Union
from PIL import Image
import torch
from transformers import pipeline
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Import our OOP base classes and decorators
from oop_examples.base_classes import MultiInheritanceModelWrapper
from oop_examples.decorators import timeit, log_exceptions, retry_on_failure



class SentimentClassifierWrapper(MultiInheritanceModelWrapper):
    """
    Wrapper for sentiment analysis models using DistilBERT.
    
    This demonstrates:
    - Text processing and sentiment classification
    - Encapsulation of NLP model complexity
    - Different input validation for text data
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        super().__init__(model_name)
        
        # Set model-specific metadata
        self.set_metadata('task', 'sentiment-analysis')
        self.set_metadata('input_type', 'text')
        self.set_metadata('output_type', 'sentiment')
        
        # Initialize model components
        self._pipeline = None
        
        # Load model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the sentiment analysis model.
        
        This demonstrates encapsulation - hiding complex initialization logic.
        """
        try:
            # Use pipeline for sentiment analysis
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self._is_loaded = True
            self.log_operation("model_initialized", None, "success")
            
        except Exception as e:
            self._increment_error_count()
            self.log_operation("model_initialization_failed", None, str(e))
            raise RuntimeError(f"Failed to initialize model {self.model_name}: {e}")
    
    @timeit
    @log_exceptions
    @retry_on_failure(max_attempts=2)
    def run(self, input_data: str) -> Dict[str, Any]:
        """
        Analyze sentiment of input text.
        
        This demonstrates:
        - Method overriding with text-specific processing
        - Consistent interface implementation
        - Specialized output formatting
        
        Args:
            input_data: Text to analyze for sentiment
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: expected non-empty string")
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        try:
            # Run sentiment analysis
            results = self._pipeline(input_data)
            
            # Handle both single result and list of results
            if isinstance(results, list):
                result = results[0]  # Take first result
            else:
                result = results
            
            # Format results consistently
            formatted_results = {
                'model_name': self.model_name,
                'task': 'sentiment-analysis',
                'input_text': input_data,
                'sentiment': result['label'],
                'confidence': round(result['score'], 4),
                'prediction': {
                    'label': result['label'],
                    'score': round(result['score'], 4),
                    'is_positive': result['label'] == 'POSITIVE'
                },
                'input_info': {
                    'type': 'text',
                    'length': len(input_data),
                    'word_count': len(input_data.split())
                }
            }
            
            return formatted_results
            
        except Exception as e:
            self._increment_error_count()
            raise RuntimeError(f"Sentiment analysis failed: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input for sentiment analysis.
        
        This overrides the parent method for text-specific validation.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid for sentiment analysis
        """
        if not isinstance(input_data, str):
            return False
        
        # Check for minimum and maximum length
        cleaned_input = input_data.strip()
        if len(cleaned_input) == 0:
            return False
        
        # Check for reasonable maximum length
        if len(cleaned_input) > 5000:  # Reasonable limit for sentiment analysis
            return False
        
        return True


class ImageClassifierWrapper(MultiInheritanceModelWrapper):
    """
    Wrapper for image classification models using Vision Transformer.
    
    This demonstrates:
    - Encapsulation of complex model initialization
    - Method overriding from base classes
    - Integration with Hugging Face transformers
    """
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__(model_name)
        
        # Set model-specific metadata
        self.set_metadata('task', 'image-classification')
        self.set_metadata('input_type', 'image')
        self.set_metadata('output_type', 'classification')
        
        # Initialize model components
        self._processor = None
        self._model = None
        self._pipeline = None
        
        # Load model asynchronously to avoid blocking UI
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the Hugging Face model components.
        
        This demonstrates encapsulation - hiding complex initialization logic.
        """
        try:
            # Use pipeline for simplicity and reliability
            self._pipeline = pipeline(
                "image-classification",
                model=self.model_name
            )
            
            self._is_loaded = True
            self.log_operation("model_initialized", None, "success")
            
        except Exception as e:
            self._increment_error_count()
            self.log_operation("model_initialization_failed", None, str(e))
            raise RuntimeError(f"Failed to initialize model {self.model_name}: {e}")
    
    @timeit
    @log_exceptions
    @retry_on_failure(max_attempts=2)
    def run(self, input_data: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Process an image through the classification model.
        
        This demonstrates:
        - Multiple decorators applied to a single method
        - Method overriding from parent class
        - Proper error handling and validation
        
        Args:
            input_data: Path to image file or PIL Image object
            
        Returns:
            Dictionary with classification results
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: expected image path or PIL Image")
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        try:
            # Handle different input types
            if isinstance(input_data, str):
                if not os.path.exists(input_data):
                    raise FileNotFoundError(f"Image file not found: {input_data}")
                image = Image.open(input_data).convert('RGB')
            elif isinstance(input_data, Image.Image):
                image = input_data.convert('RGB')
            else:
                raise TypeError("Input must be image path (str) or PIL Image")
            
            # Run inference - get top 5 predictions
            predictions = self._pipeline(image, top_k=5)
            
            # Format results for consistent output
            formatted_results = {
                'model_name': self.model_name,
                'task': 'image-classification',
                'predictions': [
                    {
                        'label': pred['label'],
                        'confidence': round(pred['score'], 4),
                        'rank': i + 1
                    }
                    for i, pred in enumerate(predictions)
                ],
                'top_prediction': {
                    'label': predictions[0]['label'],
                    'confidence': round(predictions[0]['score'], 4)
                } if predictions else None,
                'input_info': {
                    'type': 'image',
                    'size': image.size if hasattr(image, 'size') else 'unknown',
                    'mode': image.mode if hasattr(image, 'mode') else 'unknown'
                }
            }
            
            return formatted_results
            
        except Exception as e:
            self._increment_error_count()
            raise RuntimeError(f"Image classification failed: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input for image classification.
        
        This overrides the parent method to provide specific validation.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid for image classification
        """
        if input_data is None:
            return False
        
        # Check if it's a valid file path
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                return False
            
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            return any(input_data.lower().endswith(ext) for ext in valid_extensions)
        
        # Check if it's a PIL Image
        elif isinstance(input_data, Image.Image):
            return True
        
        return False
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported image formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']


class TextGeneratorWrapper(MultiInheritanceModelWrapper):
    """
    Wrapper for text generation models using GPT-based models.
    
    This demonstrates:
    - Alternative implementation of the same interface (polymorphism)
    - Different initialization and processing logic
    - Specific configuration for text generation
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        super().__init__(model_name)
        
        # Set model-specific metadata
        self.set_metadata('task', 'text-generation')
        self.set_metadata('input_type', 'text')
        self.set_metadata('output_type', 'generated_text')
        
        # Text generation specific settings
        self._max_length = 100
        self._num_return_sequences = 1
        self._temperature = 0.7
        
        # Initialize model components
        self._pipeline = None
        self._tokenizer = None
        
        # Load model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the text generation model.
        
        Demonstrates different initialization approach from ImageClassifier.
        """
        try:
            # Use pipeline for text generation
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                pad_token_id=50256  # GPT-2 specific pad token
            )
            
            self._is_loaded = True
            self.log_operation("model_initialized", None, "success")
            
        except Exception as e:
            self._increment_error_count()
            self.log_operation("model_initialization_failed", None, str(e))
            raise RuntimeError(f"Failed to initialize text generator {self.model_name}: {e}")
    
    @timeit
    @log_exceptions
    @retry_on_failure(max_attempts=2)
    def run(self, input_data: str, max_length: int = None, 
            temperature: float = None) -> Dict[str, Any]:
        """
        Generate text based on input prompt.
        
        This demonstrates:
        - Method overriding with different signature
        - Flexible parameter handling
        - Different output format from ImageClassifier
        
        Args:
            input_data: Text prompt for generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary with generated text results
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: expected non-empty string")
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        # Use provided parameters or defaults
        max_len = max_length or self._max_length
        temp = temperature if temperature is not None else self._temperature
        
        try:
            # Generate text with specified parameters
            results = self._pipeline(
                input_data,
                max_length=len(input_data.split()) + max_len,
                temperature=temp,
                num_return_sequences=self._num_return_sequences,
                do_sample=True,
                pad_token_id=self._pipeline.tokenizer.eos_token_id
            )
            
            # Format results
            formatted_results = {
                'model_name': self.model_name,
                'task': 'text-generation',
                'input_prompt': input_data,
                'generated_texts': [
                    {
                        'text': result['generated_text'],
                        'continuation': result['generated_text'][len(input_data):].strip(),
                        'sequence_id': i + 1
                    }
                    for i, result in enumerate(results)
                ],
                'primary_generation': results[0]['generated_text'] if results else '',
                'generation_params': {
                    'max_length': max_len,
                    'temperature': temp,
                    'num_sequences': self._num_return_sequences
                },
                'input_info': {
                    'type': 'text',
                    'length': len(input_data),
                    'word_count': len(input_data.split())
                }
            }
            
            return formatted_results
            
        except Exception as e:
            self._increment_error_count()
            raise RuntimeError(f"Text generation failed: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input for text generation.
        
        This overrides the parent method for text-specific validation.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid for text generation
        """
        if not isinstance(input_data, str):
            return False
        
        # Check for minimum length and content
        cleaned_input = input_data.strip()
        if len(cleaned_input) == 0:
            return False
        
        # Check for reasonable maximum length (to avoid memory issues)
        if len(cleaned_input) > 1000:  # Arbitrary limit
            return False
        
        return True
    
    def set_generation_params(self, max_length: int = None, 
                             temperature: float = None,
                             num_sequences: int = None) -> None:
        """
        Configure text generation parameters.
        
        Args:
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            num_sequences: Number of sequences to generate
        """
        if max_length is not None:
            self._max_length = max(1, min(max_length, 200))  # Reasonable bounds
        
        if temperature is not None:
            self._temperature = max(0.1, min(temperature, 1.0))  # Valid range
        
        if num_sequences is not None:
            self._num_return_sequences = max(1, min(num_sequences, 3))  # Limit for performance


class SpeechToTextWrapper(MultiInheritanceModelWrapper):
    """
    Wrapper for speech-to-text models using OpenAI Whisper.
    
    This demonstrates:
    - Audio processing capabilities
    - File handling and validation
    - Integration with audio processing libraries
    """
    
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        super().__init__(model_name)
        
        # Set model-specific metadata
        self.set_metadata('task', 'automatic-speech-recognition')
        self.set_metadata('input_type', 'audio')
        self.set_metadata('output_type', 'transcribed_text')
        
        # Initialize model components
        self._pipeline = None
        
        # Load model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the speech-to-text model.
        
        This demonstrates specialized model initialization for audio processing.
        """
        try:
            if not AUDIO_AVAILABLE:
                raise ImportError("Audio processing libraries not available. Install librosa and soundfile.")
            
            # Use pipeline for automatic speech recognition
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self._is_loaded = True
            self.log_operation("model_initialized", None, "success")
            
        except Exception as e:
            self._increment_error_count()
            self.log_operation("model_initialization_failed", None, str(e))
            raise RuntimeError(f"Failed to initialize model {self.model_name}: {e}")
    
    @timeit
    @log_exceptions
    @retry_on_failure(max_attempts=2)
    def run(self, input_data: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        This demonstrates:
        - Audio file processing using librosa (no ffmpeg required)
        - Specialized input handling for media files
        - Different output format for transcription
        
        Args:
            input_data: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: expected valid audio file path")
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        try:
            # Load audio using librosa (no ffmpeg required)
            import librosa
            
            # Load audio file - librosa handles various formats natively
            audio, sample_rate = librosa.load(input_data, sr=16000)  # Whisper expects 16kHz
            
            # Run speech recognition with raw audio array
            # Enable timestamps for long audio files (>30 seconds)
            result = self._pipeline(audio, return_timestamps=True)
            
            # Get file information
            file_size = os.path.getsize(input_data) if os.path.exists(input_data) else 0
            duration = len(audio) / sample_rate
            
            # Format results consistently
            formatted_results = {
                'model_name': self.model_name,
                'task': 'automatic-speech-recognition',
                'audio_file': os.path.basename(input_data),
                'transcribed_text': result['text'],
                'input_info': {
                    'type': 'audio',
                    'file_path': input_data,
                    'file_size_bytes': file_size,
                    'file_extension': os.path.splitext(input_data)[1].lower(),
                    'duration_seconds': round(duration, 2),
                    'sample_rate': sample_rate
                }
            }
            
            return formatted_results
            
        except Exception as e:
            self._increment_error_count()
            raise RuntimeError(f"Speech-to-text transcription failed: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input for speech-to-text processing.
        
        This overrides the parent method for audio-specific validation.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid audio file
        """
        if not isinstance(input_data, str):
            return False
        
        # Check if file exists
        if not os.path.exists(input_data):
            return False
        
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        file_ext = os.path.splitext(input_data)[1].lower()
        
        return file_ext in valid_extensions
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.wav', '.mp3', '.flac', '.m4a', '.ogg']


class TextToImageWrapper(MultiInheritanceModelWrapper):
    """
    Wrapper for text-to-image generation using Stable Diffusion.
    
    This demonstrates:
    - Advanced generative AI integration
    - Image generation from text prompts
    - Resource-intensive model handling
    """
    
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2"):
        super().__init__(model_name)
        
        # Set model-specific metadata
        self.set_metadata('task', 'text-to-image')
        self.set_metadata('input_type', 'text')
        self.set_metadata('output_type', 'generated_image')
        
        # Text-to-image specific settings
        self._num_inference_steps = 20  # Reduced for faster generation
        self._guidance_scale = 7.5
        self._image_size = 512
        
        # Initialize model components
        self._pipeline = None
        
        # Load model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """
        Initialize the text-to-image generation model.
        
        This demonstrates handling of resource-intensive models.
        """
        try:
            if not DIFFUSERS_AVAILABLE:
                raise ImportError("Diffusers library not available. Install diffusers for text-to-image generation.")
            
            # Import here to avoid issues if diffusers not installed
            from diffusers import StableDiffusionPipeline
            
            # Use lower precision for faster loading and less memory usage
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            
            # Move to appropriate device
            if torch.cuda.is_available():
                self._pipeline = self._pipeline.to("cuda")
            else:
                # Enable memory efficient attention for CPU
                self._pipeline.enable_attention_slicing()
            
            self._is_loaded = True
            self.log_operation("model_initialized", None, "success")
            
        except Exception as e:
            self._increment_error_count()
            self.log_operation("model_initialization_failed", None, str(e))
            raise RuntimeError(f"Failed to initialize model {self.model_name}: {e}")
    
    @timeit
    @log_exceptions
    @retry_on_failure(max_attempts=1)  # Only 1 attempt due to resource intensity
    def run(self, input_data: str, num_images: int = 1) -> Dict[str, Any]:
        """
        Generate image(s) from text prompt.
        
        This demonstrates:
        - Generative AI processing
        - Resource management for intensive tasks
        - Image output handling
        
        Args:
            input_data: Text prompt for image generation
            num_images: Number of images to generate (default 1)
            
        Returns:
            Dictionary with generated image results
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input: expected descriptive text prompt")
        
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call _initialize_model() first.")
        
        try:
            # Limit number of images for performance
            num_images = max(1, min(num_images, 2))
            
            # Generate image(s)
            result = self._pipeline(
                input_data,
                num_images_per_prompt=num_images,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                height=self._image_size,
                width=self._image_size
            )
            
            # Save images to temporary files and collect paths
            import tempfile
            image_paths = []
            for i, image in enumerate(result.images):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                image.save(temp_file.name)
                image_paths.append(temp_file.name)
            
            # Format results consistently
            formatted_results = {
                'model_name': self.model_name,
                'task': 'text-to-image',
                'input_prompt': input_data,
                'generated_images': [
                    {
                        'image_path': path,
                        'image_index': i + 1,
                        'size': f"{self._image_size}x{self._image_size}"
                    }
                    for i, path in enumerate(image_paths)
                ],
                'generation_params': {
                    'num_images': num_images,
                    'inference_steps': self._num_inference_steps,
                    'guidance_scale': self._guidance_scale,
                    'image_size': f"{self._image_size}x{self._image_size}"
                },
                'input_info': {
                    'type': 'text',
                    'prompt_length': len(input_data),
                    'word_count': len(input_data.split())
                }
            }
            
            return formatted_results
            
        except Exception as e:
            self._increment_error_count()
            raise RuntimeError(f"Text-to-image generation failed: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input for text-to-image generation.
        
        This overrides the parent method for image generation validation.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if input is valid for image generation
        """
        if not isinstance(input_data, str):
            return False
        
        # Check for minimum and maximum length
        cleaned_input = input_data.strip()
        if len(cleaned_input) < 3:  # Very short prompts usually don't work well
            return False
        
        if len(cleaned_input) > 1000:  # Very long prompts can cause issues
            return False
        
        return True
    
    def set_generation_params(self, inference_steps: int = None, 
                             guidance_scale: float = None,
                             image_size: int = None) -> None:
        """
        Configure image generation parameters.
        
        Args:
            inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            image_size: Size of generated image (square)
        """
        if inference_steps is not None:
            self._num_inference_steps = max(10, min(inference_steps, 50))  # Reasonable bounds
        
        if guidance_scale is not None:
            self._guidance_scale = max(1.0, min(guidance_scale, 20.0))  # Valid range
        
        if image_size is not None:
            # Ensure size is valid for the model
            valid_sizes = [256, 512, 768, 1024]
            self._image_size = min(valid_sizes, key=lambda x: abs(x - image_size))


# Factory function for creating model wrappers (demonstrates Factory pattern)
def create_model_wrapper(model_type: str, model_name: str = None) -> MultiInheritanceModelWrapper:
    """
    Factory function to create appropriate model wrapper based on type.
    
    This demonstrates:
    - Factory pattern
    - Polymorphism (all wrappers implement same interface)
    - Flexible model creation for all four AI model types
    
    Args:
        model_type: Type of model ('sentiment_classifier', 'image_classifier', 'speech_to_text', 'text_to_image')
        model_name: Specific model name (uses default if None)
        
    Returns:
        Model wrapper instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    # Import the new config structure
    from .model_info import MODELS_CONFIG
    
    if model_type == 'sentiment_classifier':
        name = model_name or MODELS_CONFIG['sentiment_classifier']['model_name']
        return SentimentClassifierWrapper(name)
    
    elif model_type == 'image_classifier':
        name = model_name or MODELS_CONFIG['image_classifier']['model_name']
        return ImageClassifierWrapper(name)
    
    elif model_type == 'speech_to_text':
        name = model_name or MODELS_CONFIG['speech_to_text']['model_name']
        return SpeechToTextWrapper(name)
    
    elif model_type == 'text_to_image':
        name = model_name or MODELS_CONFIG['text_to_image']['model_name']
        return TextToImageWrapper(name)
    
    # Legacy support for older model types
    elif model_type == 'text_generator':
        # Use sentiment classifier as it's more specific than generic text generation
        name = model_name or MODELS_CONFIG['sentiment_classifier']['model_name']
        return SentimentClassifierWrapper(name)
    
    else:
        available_types = list(MODELS_CONFIG.keys())
        raise ValueError(f"Unsupported model type '{model_type}'. "
                        f"Available types: {available_types}")


# Example usage and testing
if __name__ == "__main__":
    # Test factory pattern
    print("Testing model factory...")
    
    try:
        # Create image classifier
        img_model = create_model_wrapper('image_classifier')
        print(f"Created image classifier: {img_model.model_name}")
        print(f"Model loaded: {img_model.is_loaded()}")
        
        # Create text generator
        text_model = create_model_wrapper('text_generator')
        print(f"Created text generator: {text_model.model_name}")
        print(f"Model loaded: {text_model.is_loaded()}")
        
        # Test sentiment analysis (updated approach)
        sentiment_model = create_model_wrapper('sentiment_classifier')
        print(f"Created sentiment classifier: {sentiment_model.model_name}")
        result = sentiment_model.run("I love this new AI technology!")
        print(f"Sentiment result: {result['predicted_label']} (confidence: {result['confidence']:.2f})")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
    
    print("\nFactory testing completed - Enhanced with four model types!")