"""
Main application window for the AI Model Integration Application.

This module contains the primary application window that orchestrates all components
and demonstrates the complete integration of OOP concepts with AI model functionality.

Demonstrates: Application architecture, component integration, event handling
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Import our custom components
from gui.widgets import InputSelector, ModelSelector, ProcessingControls
from gui.ui_helpers import OutputDisplay, OOPExplanationDisplay
from models.model_factory import get_model_factory
from models.model_info import MODELS_CONFIG, USAGE_GUIDES


class AIModelApp:
    """
    Main application class that coordinates all components.
    
    This demonstrates:
    - Application architecture patterns
    - Component coordination
    - Event-driven programming
    - Threading integration for responsive UI
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.model_factory = get_model_factory()
        self.current_model = None
        self.current_model_key = None
        
        # Threading components
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_processing = False
        
        # UI Components
        self.input_selector = None
        self.model_selector = None
        self.processing_controls = None
        self.output_display = None
        self.oop_explanation = None
        
        self._setup_window()
        self._create_layout()
        self._initialize_models()
        self._start_result_monitor()
    
    def _setup_window(self):
        """Configure the main application window."""
        self.root.title("HIT137 Assignment 3 - AI Model Integration Application")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')  # Modern theme
        
        # Configure custom styles
        style.configure("Title.TLabel", font=("Arial", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Arial", 10, "italic"))
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    
    def _create_layout(self):
        """Create the main application layout."""
        # Title section
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = ttk.Label(
            title_frame,
            text="AI Model Integration Application",
            style="Title.TLabel"
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Demonstrating Advanced OOP Concepts with Hugging Face Models",
            style="Subtitle.TLabel"
        )
        subtitle_label.pack()
        
        # Main content area
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel (Input and Model Selection)
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        
        self.input_selector = InputSelector(
            left_panel,
            on_input_change=self._on_input_change
        )
        self.input_selector.pack(fill='both', expand=True, pady=(0, 5))
        
        # Get model information for selector
        # Convert MODELS_CONFIG to the format expected by widgets
        models_info = {}
        for key, config in MODELS_CONFIG.items():
            models_info[key] = {
                'name': config['model_name'],
                'display_name': config['display_name'],
                'task': config['task'],
                'description': config['description'],
                'input_type': config['input_type'],
                'output_type': config['output_type'],
                'model_size': config.get('model_size', 'Unknown'),
                'author': config.get('author', 'Unknown'),
                'license': config.get('license', 'Unknown'),
                'example_usage': config.get('example_usage', 'No example available'),
                'requirements': config.get('requirements', ['None']),
                'huggingface_url': config.get('huggingface_url', 'Not available')
            }
        self.model_selector = ModelSelector(
            left_panel,
            models_info,
            on_model_change=self._on_model_change
        )
        self.model_selector.pack(fill='both', expand=True, pady=(0, 5))
        
        # Processing controls
        self.processing_controls = ProcessingControls(
            left_panel,
            on_run=self._on_run_model,
            on_clear=self._on_clear_all
        )
        self.processing_controls.pack(fill='x', pady=(0, 5))
        
        # Right panel (Output and Information)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill='both', expand=True)
        
        # Output tab
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="📊 Results")
        
        self.output_display = OutputDisplay(output_frame)
        self.output_display.pack(fill='both', expand=True)
        
        # OOP Explanation tab
        oop_frame = ttk.Frame(self.notebook)
        self.notebook.add(oop_frame, text="🏗️ OOP Concepts")
        
        self.oop_explanation = OOPExplanationDisplay(oop_frame)
        self.oop_explanation.pack(fill='both', expand=True)
        
        # Model Info tab
        info_frame = ttk.Frame(self.notebook)
        self.notebook.add(info_frame, text="ℹ️ Model Details")
        
        self._create_model_info_tab(info_frame)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready - Select input and model to begin",
            relief='sunken'
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def _create_model_info_tab(self, parent):
        """Create the model information tab content."""
        info_text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state='disabled',
            height=20  # Fixed height to prevent UI overflow
        )
        info_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initial content
        initial_content = """🤖 AI Model Integration Application - Model Information

This application demonstrates advanced Object-Oriented Programming concepts through the integration of Hugging Face AI models.

📋 Available Models:

1️⃣ Image Classification Model
   • Model: google/vit-base-patch16-224 (Vision Transformer)
   • Task: Classify images into 1000+ ImageNet categories
   • Input: JPG, PNG, BMP, TIFF image files
   • Output: Top 5 predictions with confidence scores
   • Use Case: Object recognition, scene classification

2️⃣ Text Generation Model
   • Model: distilgpt2 (Distilled GPT-2)
   • Task: Generate creative text continuations
   • Input: Text prompts (any length)
   • Output: AI-generated text based on prompt
   • Use Case: Creative writing, text completion

🏗️ OOP Concepts Demonstrated:

✓ Multiple Inheritance: ModelWrapper classes inherit from base classes and mixins
✓ Encapsulation: Complex model operations hidden behind simple interfaces
✓ Polymorphism: All models implement the same run() method signature
✓ Method Overriding: Subclasses customize parent behavior for specific needs
✓ Multiple Decorators: @timeit, @log_exceptions, @retry_on_failure applied together

🔧 Technical Architecture:

• Factory Pattern: ModelFactory creates appropriate model instances
• Observer Pattern: UI components respond to model and input changes
• Threading: Background processing prevents UI freezing
• Error Handling: Comprehensive exception handling with user feedback
• Logging: Detailed operation logging for debugging and monitoring

📚 Educational Value:

This application serves as a comprehensive example of how advanced OOP principles
can be applied to create maintainable, extensible AI applications. Each component
demonstrates specific design patterns and programming best practices.

🚀 Getting Started:

1. Select your input type (Text or Image)
2. Provide the appropriate input data
3. Choose an AI model from the dropdown
4. Click "Run Model" to see results
5. Explore the OOP Concepts tab to understand the implementation
"""
        
        info_text.config(state='normal')
        info_text.insert('1.0', initial_content)
        info_text.config(state='disabled')
        
        self.model_info_text = info_text
    
    def _initialize_models(self):
        """Initialize the AI models in background."""
        def init_models():
            try:
                # Create default models to cache them
                self.model_factory.create_all_default_models()
                self._update_status("Models initialized successfully")
            except Exception as e:
                self._update_status(f"Model initialization warning: {e}")
        
        # Run initialization in background thread
        init_thread = threading.Thread(target=init_models, daemon=True)
        init_thread.start()
    
    def _start_result_monitor(self):
        """Start the result monitoring loop."""
        self._check_result_queue()
    
    def _check_result_queue(self):
        """Check for results from background processing."""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._handle_processing_result(result)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._check_result_queue)
    
    def _on_input_change(self, input_type: str, input_data: Any):
        """Handle input change events."""
        self._update_status(f"Input changed: {input_type}")
        
        # Filter models based on input type
        self.model_selector.filter_models_by_input_type(input_type)
        
        # Get the newly selected model after filtering
        new_model_key = self.model_selector.get_selected_model_key()
        if new_model_key and new_model_key != self.current_model_key:
            self.current_model_key = new_model_key
            display_name = MODELS_CONFIG.get(new_model_key, {}).get('display_name', new_model_key)
            self._update_status(f"Auto-selected model: {display_name}")
            
            # Update model info display
            self._update_model_info_display(new_model_key)
    
    def _on_model_change(self, model_key: str):
        """Handle model selection change."""
        self.current_model_key = model_key
        display_name = MODELS_CONFIG.get(model_key, {}).get('display_name', model_key)
        self._update_status(f"Model selected: {display_name}")
        
        # Update model info display
        self._update_model_info_display(model_key)
    
    def _update_model_info_display(self, model_key: str):
        """Update the model information tab with current model details."""
        try:
            model_info = MODELS_CONFIG.get(model_key, {})
            usage_guide = USAGE_GUIDES.get(model_key, {})
            
            detailed_info = f"""🤖 Current Model: {model_info.get('display_name', 'Unknown')}

📊 Model Specifications:
• Name: {model_info.get('model_name', 'Unknown')}
• Task: {model_info.get('task', 'unknown').replace('-', ' ').title()}
• Size: {model_info.get('model_size', 'Unknown')}
• Author: {model_info.get('author', 'Unknown')}
• License: {model_info.get('license', 'Unknown')}

📝 Description:
{model_info.get('description', 'No description available')}

💡 Usage Tips:
{chr(10).join(f"• {tip}" for tip in usage_guide.get('preparation_tips', []))}

📖 Interpreting Results:
{chr(10).join(f"• {guide}" for guide in usage_guide.get('interpretation_guide', []))}

⚠️ Limitations:
{chr(10).join(f"• {limit}" for limit in usage_guide.get('limitations', []))}

🔗 More Information:
{model_info.get('huggingface_url', 'Not available')}

📦 Requirements:
{', '.join(model_info.get('requirements', ['None']))}
"""
            
            self.model_info_text.config(state='normal')
            self.model_info_text.delete('1.0', tk.END)
            self.model_info_text.insert('1.0', detailed_info)
            self.model_info_text.config(state='disabled')
            
        except Exception as e:
            self._update_status(f"Error updating model info: {e}")
    
    def _on_run_model(self):
        """Handle run model button click."""
        if self.is_processing:
            return
        
        # Clear existing results before running new model
        if self.output_display:
            self.output_display.clear_display()
        
        # Get current input
        input_type, input_data = self.input_selector.get_current_input()
        
        if not input_data:
            messagebox.showwarning(
                "No Input",
                "Please provide input data before running the model."
            )
            return
        
        if not self.current_model_key:
            messagebox.showwarning(
                "No Model Selected",
                "Please select a model before processing."
            )
            return
        
        # Start processing in background
        self._start_background_processing(input_type, input_data, self.current_model_key)
    
    def _start_background_processing(self, input_type: str, input_data: Any, model_key: str):
        """Start model processing in background thread."""
        self.is_processing = True
        self.processing_controls.set_processing_state(True, "Processing...")
        self._update_status("Running model inference...")
        
        def process_model():
            try:
                # Create or get model instance
                model = self.model_factory.create_model(model_key)
                
                # Run inference
                result = model.run(input_data)
                
                # Add metadata
                result['processing_time'] = datetime.now().isoformat()
                result['input_type'] = input_type
                result['model_key'] = model_key
                
                # Send result to main thread
                self.result_queue.put(('success', result))
                
            except Exception as e:
                # Send error to main thread
                self.result_queue.put(('error', str(e)))
        
        # Start processing thread
        process_thread = threading.Thread(target=process_model, daemon=True)
        process_thread.start()
    
    def _handle_processing_result(self, result_data):
        """Handle processing results from background thread."""
        result_type, result_content = result_data
        
        self.is_processing = False
        self.processing_controls.set_processing_state(False)
        
        if result_type == 'success':
            # Display results
            self.output_display.display_result(result_content)
            self._update_status("Processing completed successfully")
            
            # Switch to results tab
            self.notebook.select(0)
            
        elif result_type == 'error':
            # Show error
            messagebox.showerror("Processing Error", f"Model processing failed:\n\n{result_content}")
            self._update_status(f"Processing failed: {result_content}")
    
    def _on_clear_all(self):
        """Handle clear all button click."""
        if self.is_processing:
            return
        
        # Clear input
        self.input_selector.clear_input()
        
        # Clear output
        self.output_display.clear_display()
        
        self._update_status("All data cleared")
    
    def _update_status(self, message: str):
        """Update the status bar."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_bar.config(text=f"[{timestamp}] {message}")


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    app = AIModelApp(root)
    root.mainloop()