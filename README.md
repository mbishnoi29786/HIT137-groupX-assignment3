# HIT137 Group Assignment 3 - AI Model Integration Application

## Project Overview
This application demonstrates advanced Object-Oriented Programming concepts while integrating Hugging Face AI models through a user-friendly Tkinter GUI. The application supports multiple input types (text, image, audio) and showcases two different AI models for various tasks.

## Features
- **Multi-input Support**: Handle text, image, and audio inputs
- **AI Model Integration**: Two Hugging Face models (Image Classification & Text Generation)
- **Advanced OOP Concepts**: Multiple inheritance, polymorphism, encapsulation, decorators
- **User-friendly GUI**: Intuitive Tkinter interface with tabbed output display
- **Real-time Processing**: Threaded model execution to prevent UI freezing

## Project Structure
```
HIT137-Assignment3/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── github_link.txt        # Repository URL
├── gui/                   # GUI components
│   ├── app_window.py      # Main application window
│   ├── widgets.py         # Custom UI widgets
│   └── ui_helpers.py      # UI utility functions
├── models/                # AI model integration
│   ├── hf_integration.py  # Hugging Face model wrappers
│   ├── model_info.py      # Model metadata
│   └── model_factory.py   # Model creation factory
├── oop_examples/          # OOP demonstrations
│   ├── base_classes.py    # Base classes and interfaces
│   └── decorators.py      # Custom decorators
└── tests/                 # Test files
    └── simple_tests.py    # Unit tests
```

## OOP Concepts Demonstrated

### 1. Multiple Inheritance
- `LoggedModelWrapper` class inherits from both `ModelLoggingMixin` and `BaseModelWrapper`
- Combines logging functionality with base model interface

### 2. Encapsulation
- Model wrappers hide complex Hugging Face pipeline internals
- Private methods and properties protect internal state

### 3. Polymorphism
- All model wrappers implement the same `run()` interface
- GUI can call any model uniformly without knowing its type

### 4. Method Overriding
- Child classes override parent methods to provide specific implementations
- Custom behavior while maintaining the same interface

### 5. Multiple Decorators
- `@timeit` and `@log_exceptions` decorators applied to multiple methods
- Demonstrates decorator stacking and reusability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mbishnoi29786/HIT137-groupX-assignment3.git
cd HIT137-Assignment3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

1. **Select Input Type**: Choose between Text, Image, or Audio
2. **Provide Input**: Enter text, select image file, or choose audio file
3. **Select Model**: Choose from available AI models
4. **Run Processing**: Click "Run" to process input with selected model
5. **View Results**: Check output in the Results tab
6. **Explore OOP**: Review OOP implementation details in the OOP Explanations tab
7. **Model Info**: Learn about the AI models in the Model Info tab

## AI Models Used

### 1. Image Classification Model
- **Model**: google/vit-base-patch16-224
- **Task**: Image classification
- **Description**: Classifies images into predefined categories
- **Input**: Image files (JPG, PNG, etc.)
- **Output**: Classification labels with confidence scores

### 2. Text Generation Model
- **Model**: distilgpt2
- **Task**: Text generation
- **Description**: Generates coherent text based on input prompts
- **Input**: Text prompts
- **Output**: Generated text continuations

## Team Members
- Manish S393232
- 

## Development Process
This project follows best practices for collaborative development:
- Feature branching with pull requests
- Code reviews and testing
- Clear commit messages and documentation
- Modular design for maintainability

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## License
This project is for educational purposes as part of HIT137 coursework.