"""
UI helper classes and components for the AI Model Integration Application.

This module provides specialized UI components for displaying results,
explanations, and managing complex UI interactions.

Demonstrates: Specialized UI components, data presentation, user experience design
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Dict, Any, List
import json
from datetime import datetime
from PIL import Image, ImageTk


class OutputDisplay(ttk.Frame):
    """
    Specialized widget for displaying AI model results in various formats.
    
    This demonstrates:
    - Dynamic content rendering
    - Multi-format result display
    - Professional result presentation
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.current_result = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the output display interface.""" 
        # Header section
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        self.result_title = ttk.Label(
            header_frame,
            text="Model Results",
            font=("Arial", 12, "bold")
        )
        self.result_title.pack(side='left')
        
        self.export_button = ttk.Button(
            header_frame,
            text="📁 Export Results",
            command=self._export_results,
            state='disabled'
        )
        self.export_button.pack(side='right')
        
        # Main content area with scrolling
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Default empty state
        self._show_empty_state()
    
    def _show_empty_state(self):
        """Display empty state when no results are available."""
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        empty_label = ttk.Label(
            self.content_frame,
            text=" Ready for AI Magic!\n\n"
                 "Select your input type, provide data, choose a model,\n"
                 "and click 'Run Model' to see results here.\n\n"
                 "Results will be displayed in an easy-to-read format\n"
                 "with detailed information about the AI's predictions.",
            justify='center',
            font=("TkDefaultFont", 10)
        )
        empty_label.pack(expand=True, pady=50)
        
        self.export_button.config(state='disabled')
    
    def display_result(self, result: Dict[str, Any]):
        """
        Display model results in appropriate format.
        
        Args:
            result: Dictionary containing model results
        """
        self.current_result = result
        
        # Clear current content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Create scrollable frame with fixed height to ensure run button visibility
        canvas = tk.Canvas(self.content_frame, height=400)  # Fixed max height
        scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Pack scrolling components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Display results based on task type
        task = result.get('task', 'unknown')
        
        if task == 'image-classification':
            self._display_image_classification_results(scrollable_frame, result)
        elif task == 'text-generation':
            self._display_text_generation_results(scrollable_frame, result)
        else:
            self._display_generic_results(scrollable_frame, result)
        
        # Update export button
        self.export_button.config(state='normal')
        
        # Update title
        model_name = result.get('model_name', 'Unknown Model')
        self.result_title.config(text=f"Results: {model_name}")
    
    def _display_image_classification_results(self, parent, result: Dict[str, Any]):
        """Display image classification results."""
        # Result summary
        summary_frame = ttk.LabelFrame(parent, text="🏷️ Classification Results")
        summary_frame.pack(fill='x', padx=5, pady=5)
        
        predictions = result.get('predictions', [])
        top_pred = result.get('top_prediction', {})
        
        if top_pred:
            # Top prediction highlight
            top_frame = ttk.Frame(summary_frame)
            top_frame.pack(fill='x', padx=10, pady=10)
            
            ttk.Label(
                top_frame,
                text="Top Prediction:",
                font=("Arial", 11, "bold")
            ).pack(anchor='w')
            
            ttk.Label(
                top_frame,
                text=f"{top_pred['label']} ({top_pred['confidence']:.1%} confidence)",
                font=("Arial", 12, "bold"),
                foreground='darkgreen'
            ).pack(anchor='w', padx=20)
        
        # All predictions table
        if predictions:
            pred_frame = ttk.LabelFrame(parent, text="All Predictions (Top 5)")
            pred_frame.pack(fill='x', padx=5, pady=5)
            
            # Create table
            columns = ('Rank', 'Label', 'Confidence', 'Percentage')
            tree = ttk.Treeview(pred_frame, columns=columns, show='headings', height=6)
            
            # Configure columns
            tree.heading('Rank', text='Rank')
            tree.heading('Label', text='Predicted Label')
            tree.heading('Confidence', text='Confidence')
            tree.heading('Percentage', text='Percentage')
            
            tree.column('Rank', width=50, anchor='center')
            tree.column('Label', width=200, anchor='w')
            tree.column('Confidence', width=80, anchor='center')
            tree.column('Percentage', width=80, anchor='center')
            
            # Add data
            for pred in predictions:
                tree.insert('', 'end', values=(
                    pred['rank'],
                    pred['label'],
                    f"{pred['confidence']:.4f}",
                    f"{pred['confidence']:.1%}"
                ))
            
            tree.pack(fill='x', padx=10, pady=10)
        
        # Input information
        input_info = result.get('input_info', {})
        if input_info:
            info_frame = ttk.LabelFrame(parent, text="📷 Input Information")
            info_frame.pack(fill='x', padx=5, pady=5)
            
            info_text = f"""Image Type: {input_info.get('type', 'Unknown')}
Image Size: {input_info.get('size', 'Unknown')}
Color Mode: {input_info.get('mode', 'Unknown')}"""
            
            ttk.Label(
                info_frame,
                text=info_text,
                font=("Consolas", 9),
                justify='left'
            ).pack(anchor='w', padx=10, pady=10)
    
    def _display_text_generation_results(self, parent, result: Dict[str, Any]):
        """Display text generation results."""
        # Input prompt
        prompt = result.get('input_prompt', '')
        if prompt:
            prompt_frame = ttk.LabelFrame(parent, text="💭 Your Prompt")
            prompt_frame.pack(fill='x', padx=5, pady=5)
            
            prompt_text = scrolledtext.ScrolledText(
                prompt_frame,
                height=3,
                wrap=tk.WORD,
                font=("Arial", 10),
                state='disabled'
            )
            prompt_text.pack(fill='x', padx=10, pady=10)
            
            prompt_text.config(state='normal')
            prompt_text.insert('1.0', prompt)
            prompt_text.config(state='disabled')
        
        # Generated text
        generated_texts = result.get('generated_texts', [])
        if generated_texts:
            for i, gen_text in enumerate(generated_texts):
                gen_frame = ttk.LabelFrame(parent, text=f"🤖 AI Generated Text #{gen_text['sequence_id']}")
                gen_frame.pack(fill='x', padx=5, pady=5)
                
                # Full generated text
                full_text = scrolledtext.ScrolledText(
                    gen_frame,
                    height=6,
                    wrap=tk.WORD,
                    font=("Arial", 10),
                    state='disabled'
                )
                full_text.pack(fill='x', padx=10, pady=10)
                
                full_text.config(state='normal')
                full_text.insert('1.0', gen_text['text'])
                full_text.config(state='disabled')
                
                # Highlight the new part (continuation only)
                if gen_text.get('continuation'):
                    cont_frame = ttk.LabelFrame(gen_frame, text="✨ New Content Only")
                    cont_frame.pack(fill='x', padx=10, pady=5)
                    
                    cont_text = scrolledtext.ScrolledText(
                        cont_frame,
                        height=4,
                        wrap=tk.WORD,
                        font=("Arial", 10),
                        state='disabled',
                        background='#f0f8ff'
                    )
                    cont_text.pack(fill='x', padx=5, pady=5)
                    
                    cont_text.config(state='normal')
                    cont_text.insert('1.0', gen_text['continuation'])
                    cont_text.config(state='disabled')
        
        # Generation parameters
        gen_params = result.get('generation_params', {})
        if gen_params:
            params_frame = ttk.LabelFrame(parent, text="⚙️ Generation Settings")
            params_frame.pack(fill='x', padx=5, pady=5)
            
            params_text = f"""Max Length: {gen_params.get('max_length', 'Unknown')} tokens
Temperature: {gen_params.get('temperature', 'Unknown')} (creativity level)
Sequences: {gen_params.get('num_sequences', 'Unknown')} generated"""
            
            ttk.Label(
                params_frame,
                text=params_text,
                font=("Consolas", 9),
                justify='left'
            ).pack(anchor='w', padx=10, pady=10)
    
    def _display_generic_results(self, parent, result: Dict[str, Any]):
        """Display generic results for unknown task types."""
        generic_frame = ttk.LabelFrame(parent, text="📋 Raw Results")
        generic_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        result_text = scrolledtext.ScrolledText(
            generic_frame,
            wrap=tk.WORD,
            font=("Consolas", 9)
        )
        result_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Format JSON nicely
        try:
            formatted_result = json.dumps(result, indent=2, ensure_ascii=False)
            result_text.insert('1.0', formatted_result)
        except Exception:
            result_text.insert('1.0', str(result))
        
        result_text.config(state='disabled')
    
    def _export_results(self):
        """Export current results to a file."""
        if not self.current_result:
            return
        
        from tkinter import filedialog
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"ai_results_{timestamp}.json"
        
        filename = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialfile=default_filename
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Export Successful", f"Results exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
    
    def clear_display(self):
        """Clear the current display."""
        self.current_result = None
        self._show_empty_state()


class OOPExplanationDisplay(ttk.Frame):
    """
    Widget for displaying OOP concept explanations and code examples.
    
    This demonstrates:
    - Educational content presentation
    - Code highlighting and formatting
    - Interactive learning support
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._setup_ui()
        self._load_explanations()
    
    def _setup_ui(self):
        """Set up the explanation display interface."""
        # Header
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=5, pady=5)
        
        title_label = ttk.Label(
            header_frame,
            text="🏗️ Object-Oriented Programming Concepts Demonstrated",
            font=("Arial", 12, "bold")
        )
        title_label.pack(side='left')
        
        # Content area with fixed height to prevent UI overflow
        self.content_text = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state='disabled',
            height=25  # Fixed height to ensure run button remains visible
        )
        self.content_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _load_explanations(self):
        """Load OOP concept explanations and examples."""
        explanations = """ OBJECT-ORIENTED PROGRAMMING CONCEPTS IN THIS APPLICATION

This application demonstrates advanced OOP concepts through practical AI model integration.
Each concept is implemented across multiple files with real-world applications.

═══════════════════════════════════════════════════════════════════════════════

1. MULTIPLE INHERITANCE
    Location: oop_examples/base_classes.py (Lines 185-220)
    Implementation: MultiInheritanceModelWrapper class

class MultiInheritanceModelWrapper(BaseModelWrapper, ModelLoggingMixin, ModelMetadataMixin):
    def __init__(self, model_name: str):
        # Call all parent constructors properly
        BaseModelWrapper.__init__(self, model_name)
        ModelLoggingMixin.__init__(self)
        ModelMetadataMixin.__init__(self)

 WHY IT'S USEFUL:
• Combines base model functionality with logging and metadata capabilities
• Demonstrates proper constructor chaining in multiple inheritance
• Solves the "diamond problem" through careful design
• Allows mixing different behaviors without code duplication

═══════════════════════════════════════════════════════════════════════════════

2. ENCAPSULATION
    Location: models/hf_integration.py (Lines 50-120)
    Implementation: ImageClassifierWrapper and TextGeneratorWrapper classes

class ImageClassifierWrapper(MultiInheritanceModelWrapper):
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        super().__init__(model_name)
        # Private attributes (indicated by underscore)
        self._processor = None
        self._model = None
        self._pipeline = None
        self._initialize_model()  # Private method
    
    def _initialize_model(self) -> None:
        '''Hide complex initialization from users'''
        # Complex Hugging Face setup hidden here
        pass

<<<<<<< HEAD
 WHY IT'S USEFUL:
=======
WHY IT'S USEFUL:
>>>>>>> 74ce47def5be02872ac62525cc6382715ec5f421
• Hides complex Hugging Face pipeline setup from users
• Protects internal state with private attributes
• Provides simple public interface (run, validate_input, etc.)
• Prevents users from breaking internal model state

═══════════════════════════════════════════════════════════════════════════════

3. POLYMORPHISM
    Location: models/model_factory.py (Lines 150-180)
    Implementation: All model wrappers implement BaseModelInterface

# All models can be used the same way regardless of type:
def process_with_any_model(model: BaseModelInterface, data: Any):
    return model.run(data)  # Works with ANY model type!

# Usage examples:
image_model = ImageClassifierWrapper("vit-model")
text_model = TextGeneratorWrapper("gpt2-model")

result1 = process_with_any_model(image_model, "cat.jpg")     # Image processing
result2 = process_with_any_model(text_model, "Hello world")  # Text processing

 WHY IT'S USEFUL:
• Same interface works with different model types
• Easy to add new model types without changing existing code
• GUI can handle any model type uniformly
• Reduces code duplication and improves maintainability

═══════════════════════════════════════════════════════════════════════════════

4. METHOD OVERRIDING
    Location: models/hf_integration.py (Lines 160-200, 280-320)
    Implementation: Different validation methods for different model types

class ImageClassifierWrapper(MultiInheritanceModelWrapper):
    def validate_input(self, input_data: Any) -> bool:
        '''Override for image-specific validation'''
        if isinstance(input_data, str):
            # Check file extension for images
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            return any(input_data.lower().endswith(ext) for ext in valid_extensions)
        return isinstance(input_data, Image.Image)

class TextGeneratorWrapper(MultiInheritanceModelWrapper):
    def validate_input(self, input_data: Any) -> bool:
        '''Override for text-specific validation'''
        if not isinstance(input_data, str):
            return False
        # Check length constraints
        return 0 < len(input_data.strip()) <= 1000

 WHY IT'S USEFUL:
• Each model type has specialized validation logic
• Maintains the same interface while providing specific behavior
• Parent class defines the contract, children implement details
• Enables type-specific error handling and user feedback

═══════════════════════════════════════════════════════════════════════════════

5. MULTIPLE DECORATORS
    Location: oop_examples/decorators.py (Lines 30-90)
    Implementation: Stacked decorators on model methods

@timeit                    # Measures execution time
@log_exceptions           # Logs any errors that occur
@retry_on_failure(max_attempts=2)  # Retries on failure
def run(self, input_data: Any) -> Dict[str, Any]:
    '''This method has THREE decorators working together!'''
    # If this fails, it will:
    # 1. Retry up to 2 times (retry_on_failure)
    # 2. Log any exceptions (log_exceptions) 
    # 3. Measure how long it takes (timeit)
    return self._process_input(input_data)

 WHY IT'S USEFUL:
• Separates cross-cutting concerns (timing, logging, retry logic)
• Decorators can be reused on multiple methods
• Clean separation of business logic from infrastructure code
• Easy to add/remove functionality by changing decorators

═══════════════════════════════════════════════════════════════════════════════

 DESIGN PATTERNS DEMONSTRATED:

 FACTORY PATTERN (models/model_factory.py):
   Creates appropriate model instances without specifying exact classes

 OBSERVER PATTERN (gui/app_window.py):
   UI components respond to input and model changes

 TEMPLATE METHOD PATTERN (oop_examples/base_classes.py):
   Base classes define algorithm structure, subclasses fill in details

 STRATEGY PATTERN (models/hf_integration.py):
   Different model implementations can be swapped interchangeably

═══════════════════════════════════════════════════════════════════════════════

 EDUCATIONAL TAKEAWAYS:

 Multiple inheritance can be powerful when used carefully
 Encapsulation hides complexity and protects internal state
 Polymorphism enables flexible, extensible code architectures
 Method overriding allows specialization while maintaining contracts
 Decorators provide clean separation of concerns

This application demonstrates how these concepts work together to create
maintainable, extensible software that can grow and adapt over time.

The key is not just using these concepts, but understanding WHEN and WHY
to use them to solve real problems in software development.
"""
        
        self.content_text.config(state='normal')
        self.content_text.insert('1.0', explanations)
        self.content_text.config(state='disabled')


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("UI Helpers Test")
    root.geometry("800x600")
    
    # Test output display
    output_display = OutputDisplay(root)
    output_display.pack(fill='both', expand=True)
    
    # Test with sample data
    sample_result = {
        'task': 'text-generation',
        'model_name': 'test-model',
        'input_prompt': 'Test prompt',
        'generated_texts': [
            {
                'text': 'Test prompt and generated continuation',
                'continuation': 'and generated continuation',
                'sequence_id': 1
            }
        ]
    }
    
    output_display.display_result(sample_result)
    
    root.mainloop()