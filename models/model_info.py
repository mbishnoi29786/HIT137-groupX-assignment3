"""
Model information and metadata for the AI Model Integration Application.

This module contains metadata about the Hugging Face models used in the application,
including model descriptions, requirements, and usage information.

Demonstrates: Data encapsulation, configuration management
"""

from typing import Dict, List, Any

# Enhanced model configuration for four AI model types - Single source of truth
MODELS_CONFIG = {
    'sentiment_classifier': {
        'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
        'display_name': 'DistilBERT Sentiment Classifier',
        'task': 'text-classification',
        'description': 'A fine-tuned DistilBERT model for sentiment analysis, optimized for binary sentiment classification (positive/negative). Based on the Stanford Sentiment Treebank dataset.',
        'input_type': 'text',
        'output_type': 'sentiment_classification',
        'model_size': 'Distilled (66M parameters)',
        'author': 'Hugging Face',
        'license': 'Apache-2.0',
        'example_usage': 'Enter text to analyze sentiment (positive/negative) with confidence scores.',
        'requirements': ['transformers', 'torch'],
        'supported_formats': ['text/plain'],
        'huggingface_url': 'https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english'
    },
    'image_classifier': {
        'model_name': 'google/vit-base-patch16-224',
        'display_name': 'Vision Transformer (Image Classifier)',
        'task': 'image-classification',
        'description': 'A Vision Transformer model trained for image classification. Can classify images into 1000+ categories from ImageNet.',
        'input_type': 'image',
        'output_type': 'classification_labels',
        'model_size': 'Base (86M parameters)',
        'author': 'Google Research',
        'license': 'Apache-2.0',
        'example_usage': 'Upload an image to get classification predictions with confidence scores.',
        'requirements': ['transformers', 'torch', 'torchvision', 'pillow'],
        'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        'huggingface_url': 'https://huggingface.co/google/vit-base-patch16-224'
    },
    'speech_to_text': {
        'model_name': 'openai/whisper-tiny',
        'display_name': 'Whisper Tiny (Speech-to-Text)',
        'task': 'automatic-speech-recognition',
        'description': 'OpenAI Whisper tiny model for speech recognition. Fast and lightweight, perfect for converting audio to text.',
        'input_type': 'audio',
        'output_type': 'transcribed_text',
        'model_size': 'Tiny (39M parameters)',
        'author': 'OpenAI',
        'license': 'MIT',
        'example_usage': 'Upload an audio file to get text transcription.',
        'requirements': ['transformers', 'torch', 'librosa', 'soundfile'],
        'supported_formats': ['.wav', '.mp3', '.flac', '.m4a'],
        'huggingface_url': 'https://huggingface.co/openai/whisper-tiny'
    },
    'text_to_image': {
        'model_name': 'stabilityai/stable-diffusion-2',
        'display_name': 'Stable Diffusion 2 (Text-to-Image)',
        'task': 'text-to-image',
        'description': 'Stable Diffusion model for generating images from text prompts. Creates high-quality images based on descriptive text input.',
        'input_type': 'text',
        'output_type': 'generated_image',
        'model_size': 'Large (865M parameters)',
        'author': 'Stability AI',
        'license': 'CreativeML Open RAIL++-M',
        'example_usage': 'Enter a descriptive text prompt to generate corresponding images.',
        'requirements': ['transformers', 'torch', 'diffusers', 'pillow'],
        'supported_formats': ['text/plain'],
        'huggingface_url': 'https://huggingface.co/stabilityai/stable-diffusion-2'
    }
}

# Detailed usage guides for each model type
USAGE_GUIDES = {
    'sentiment_classifier': {
        'preparation_tips': [
            'Works best with complete sentences and clear opinions',
            'Can handle informal language and social media text',
            'Longer texts may provide more reliable sentiment scores',
            'Works primarily with English text'
        ],
        'interpretation_guide': [
            'POSITIVE: Text expresses positive sentiment, emotions, or opinions',
            'NEGATIVE: Text expresses negative sentiment, emotions, or opinions',
            'Confidence score ranges from 0.5 to 1.0 (higher is more certain)',
            'Scores close to 0.5 indicate neutral or ambiguous sentiment'
        ],
        'limitations': [
            'Trained primarily on English text',
            'May struggle with sarcasm or complex irony',
            'Context-dependent meanings might be misinterpreted',
            'Very short texts may have less reliable predictions'
        ]
    },
    'image_classifier': {
        'preparation_tips': [
            'Use clear, well-lit images for better results',
            'Single-object images work better than complex scenes',
            'Standard photo formats: JPEG, PNG, GIF are supported',
            'Images are automatically resized to 224x224 pixels'
        ],
        'interpretation_guide': [
            'Top predictions show the most likely object categories',
            'Confidence scores indicate model certainty (0-100%)',
            'Multiple predictions help understand scene complexity',
            'ImageNet categories include 1000 common objects and animals'
        ],
        'limitations': [
            'Limited to ImageNet categories (no custom objects)',
            'May struggle with abstract art or unusual angles',
            'Performance varies with image quality and lighting',
            'Cannot identify multiple objects in complex scenes accurately'
        ]
    },
    'speech_to_text': {
        'preparation_tips': [
            'Clear audio with minimal background noise works best',
            'Supported formats: WAV, MP3, FLAC, M4A, OGG',
            'Shorter audio clips (under 30 seconds) process faster',
            'Multiple languages supported, but English works best'
        ],
        'interpretation_guide': [
            'Transcribed text appears exactly as spoken',
            'Punctuation is automatically added',
            'Capitalization follows standard rules',
            'Processing time depends on audio length'
        ],
        'limitations': [
            'Background noise can affect accuracy',
            'Heavy accents may reduce transcription quality',
            'Very quiet or distorted audio may fail',
            'Large audio files may require more processing time'
        ]
    },
    'text_to_image': {
        'preparation_tips': [
            'Be descriptive and specific in your prompts',
            'Include style keywords like "photorealistic", "artistic", "painting"',
            'Mention colors, lighting, and composition preferences',
            'Shorter prompts (under 100 words) often work better'
        ],
        'interpretation_guide': [
            'Generated images are 512x512 pixels by default',
            'Multiple images may be generated for variety',
            'Results vary each time due to random generation process',
            'Complex prompts may not always be fully interpreted'
        ],
        'limitations': [
            'Very resource-intensive - requires significant GPU/CPU time',
            'Cannot guarantee exact reproduction of described scenes',
            'May have difficulty with text within images',
            'Human faces and hands may appear distorted',
            'Takes several minutes to generate images'
        ]
    }
}


class ModelInfo:
    """
    Container class for model metadata and information.
    
    This demonstrates:
    - Data encapsulation
    - Information hiding
    - Configuration management
    
    Uses the module-level MODELS_CONFIG for single source of truth.
    """
    
    @classmethod
    def get_model_info(cls, model_key: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific model.
        
        Args:
            model_key: Key identifying the model (e.g., 'image_classifier')
            
        Returns:
            Dictionary containing model information
            
        Raises:
            KeyError: If model_key is not found
        """
        if model_key not in MODELS_CONFIG:
            available_keys = list(MODELS_CONFIG.keys())
            raise KeyError(f"Model '{model_key}' not found. Available models: {available_keys}")
        
        return MODELS_CONFIG[model_key].copy()
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available models.
        
        Returns:
            Dictionary mapping model keys to their information
        """
        return MODELS_CONFIG.copy()
    
    @classmethod
    def get_model_names(cls) -> List[str]:
        """
        Get list of all available model keys.
        
        Returns:
            List of model keys
        """
        return list(MODELS_CONFIG.keys())
    
    @classmethod
    def get_models_by_input_type(cls, input_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get models that support a specific input type.
        
        Args:
            input_type: Type of input (e.g., 'image', 'text', 'audio')
            
        Returns:
            Dictionary of models supporting the input type
        """
        matching_models = {}
        
        for key, config in MODELS_CONFIG.items():
            if config.get('input_type') == input_type:
                matching_models[key] = config.copy()
        
        return matching_models
    
    @classmethod
    def get_model_requirements(cls, model_key: str) -> List[str]:
        """
        Get the Python package requirements for a specific model.
        
        Args:
            model_key: Key identifying the model
            
        Returns:
            List of required Python packages
        """
        model_info = cls.get_model_info(model_key)
        return model_info.get('requirements', [])
    
    @classmethod
    def get_display_name(cls, model_key: str) -> str:
        """
        Get the user-friendly display name for a model.
        
        Args:
            model_key: Key identifying the model
            
        Returns:
            Display name for the model
        """
        model_info = cls.get_model_info(model_key)
        return model_info.get('display_name', model_key)
    
    @classmethod
    def validate_model_key(cls, model_key: str) -> bool:
        """
        Check if a model key is valid.
        
        Args:
            model_key: Key to validate
            
        Returns:
            True if the key is valid, False otherwise
        """
        return model_key in MODELS_CONFIG
    
    @classmethod
    def get_supported_input_types(cls) -> List[str]:
        """
        Get all supported input types across all models.
        
        Returns:
            List of unique input types
        """
        input_types = set()
        for config in MODELS_CONFIG.values():
            input_types.add(config.get('input_type', 'unknown'))
        
        return sorted(list(input_types))


class ModelUsageGuide:
    """
    Provides usage guidelines and best practices for each model.
    
    This demonstrates:
    - Documentation encapsulation
    - User guidance systems
    - Best practices organization
    
    Uses the module-level USAGE_GUIDES for single source of truth.
    """
    
    @classmethod
    def get_usage_guide(cls, model_key: str) -> Dict[str, List[str]]:
        """
        Get usage guide for a specific model.
        
        Args:
            model_key: Key identifying the model
            
        Returns:
            Dictionary with preparation tips, interpretation guide, and limitations
        """
        if model_key not in USAGE_GUIDES:
            return {
                'preparation_tips': ['No specific guidance available'],
                'interpretation_guide': ['Refer to model documentation'],
                'limitations': ['Unknown limitations - use with caution']
            }
        
        return USAGE_GUIDES[model_key].copy()
    
    @classmethod
    def get_all_guides(cls) -> Dict[str, Dict[str, List[str]]]:
        """
        Get usage guides for all models.
        
        Returns:
            Dictionary mapping model keys to their usage guides
        """
        return USAGE_GUIDES.copy()


# Example usage and validation
if __name__ == "__main__":
    # Demonstrate model information access
    print("Available models:", ModelInfo.get_model_names())
    
    # Get specific model info
    img_model = ModelInfo.get_model_info('image_classifier')
    print("\nImage Classifier Info:")
    print(f"Name: {img_model['model_name']}")
    print(f"Description: {img_model['description']}")
    
    # Get usage guide
    guide = ModelUsageGuide.get_usage_guide('image_classifier')
    print(f"\nUsage Tips: {guide['preparation_tips'][:2]}")
    
    # Demonstrate input type filtering
    text_models = ModelInfo.get_models_by_input_type('text')
    print(f"\nText models: {list(text_models.keys())}")