"""
Custom UI widgets for the AI Model Integration Application.

This module provides reusable custom widgets that extend Tkinter functionality
and demonstrate object-oriented GUI design patterns.

Demonstrates: Widget composition, event handling, custom UI components
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter import font as tkFont
from typing import Dict, Any, Callable, Optional, List
from PIL import Image, ImageTk
import os
import threading

class InputSelector(ttk.LabelFrame):
    """
    Custom widget for selecting input type and providing appropriate input interface.
    
    This demonstrates:
    - Widget composition
    - Event-driven programming
    - Dynamic interface updates
    """
    
    def __init__(self, parent, on_input_change: Callable = None, **kwargs):
        super().__init__(parent, text="Input Selection", **kwargs)
        
        self.on_input_change = on_input_change
        self.current_input_type = tk.StringVar(value="text")
        self.current_input_data = None
        
        self._setup_ui()
        self._bind_events()
    
    def _setup_ui(self):
        """Set up the user interface elements."""
        # Input type selector
        type_frame = ttk.Frame(self)
        type_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(type_frame, text="Input Type:").pack(side='left')
        
        self.type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.current_input_type,
            values=["text", "image", "audio"],
            state="readonly",
            width=10
        )
        self.type_combo.pack(side='left', padx=(5, 0))
        
        # Dynamic input container
        self.input_container = ttk.Frame(self)
        self.input_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initialize with text input
        self._create_text_input()
    
    def _bind_events(self):
        """Bind event handlers."""
        self.type_combo.bind('<<ComboboxSelected>>', self._on_type_change)
    
    def _on_type_change(self, event=None):
        """Handle input type change."""
        # Clear current input container
        for widget in self.input_container.winfo_children():
            widget.destroy()
        
        input_type = self.current_input_type.get()
        
        # Create appropriate input widget
        if input_type == "text":
            self._create_text_input()
        elif input_type == "image":
            self._create_image_input()
        elif input_type == "audio":
            self._create_audio_input()
        
        # Notify parent of change
        if self.on_input_change:
            self.on_input_change(input_type, None)
    
    def _create_text_input(self):
        """Create text input widget."""
        text_frame = ttk.LabelFrame(self.input_container, text="Text Input")
        text_frame.pack(fill='both', expand=True)
        
        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            height=6,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.text_widget.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add placeholder text
        placeholder = "Enter your text prompt here."
        self.text_widget.insert('1.0', placeholder)
        self.text_widget.config(fg='gray')
        
        # Bind focus events for placeholder behavior
        self.text_widget.bind('<FocusIn>', self._on_text_focus_in)
        self.text_widget.bind('<FocusOut>', self._on_text_focus_out)
        self.text_widget.bind('<KeyRelease>', self._on_text_change)
    
    def _create_image_input(self):
        """Create image input widget."""
        image_frame = ttk.LabelFrame(self.input_container, text="Image Input")
        image_frame.pack(fill='both', expand=True)
        
        # File selection frame
        file_frame = ttk.Frame(image_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        self.image_path_var = tk.StringVar()
        
        ttk.Label(file_frame, text="Image File:").pack(side='left')
        
        path_entry = ttk.Entry(
            file_frame,
            textvariable=self.image_path_var,
            state='readonly'
        )
        path_entry.pack(side='left', fill='x', expand=True, padx=(5, 5))
        
        browse_btn = ttk.Button(
            file_frame,
            text="Browse...",
            command=self._browse_image_file
        )
        browse_btn.pack(side='right')
        
        # Image preview frame
        preview_frame = ttk.LabelFrame(image_frame, text="Preview")
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.image_preview = ttk.Label(
            preview_frame,
            text="No image selected",
            anchor='center'
        )
        self.image_preview.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _create_audio_input(self):
        """Create audio input widget with file selection and playback capability."""
        audio_frame = ttk.LabelFrame(self.input_container, text="Audio Input")
        audio_frame.pack(fill='both', expand=True)
        
        # File selection frame
        file_frame = ttk.Frame(audio_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        self.audio_path_var = tk.StringVar()
        
        ttk.Label(file_frame, text="Audio File:").pack(side='left')
        
        path_entry = ttk.Entry(
            file_frame,
            textvariable=self.audio_path_var,
            state='readonly'
        )
        path_entry.pack(side='left', fill='x', expand=True, padx=(5, 5))
        
        browse_btn = ttk.Button(
            file_frame,
            text="Browse...",
            command=self._browse_audio_file
        )
        browse_btn.pack(side='right')
        
        # Audio controls frame
        controls_frame = ttk.Frame(audio_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        self.play_btn = ttk.Button(
            controls_frame,
            text="▶ Play Audio",
            command=self._play_audio,
            state='disabled'
        )
        self.play_btn.pack(side='left', padx=(0, 5))
        
        self.stop_btn = ttk.Button(
            controls_frame,
            text="⏹ Stop",
            command=self._stop_audio,
            state='disabled'
        )
        self.stop_btn.pack(side='left')
        
        # Audio info frame
        info_frame = ttk.LabelFrame(audio_frame, text="Audio Information")
        info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.audio_info = ttk.Label(
            info_frame,
            text="No audio file selected\nSupported formats: WAV (.wav), MP3 (.mp3), FLAC (.flac), M4A (.m4a), OGG (.ogg)",
            justify='left',
            anchor='nw'
        )
        self.audio_info.pack(fill='both', expand=True, padx=5, pady=5)
    
    def _on_text_focus_in(self, event):
        """Handle text widget focus in (remove placeholder)."""
        if self.text_widget.get('1.0', tk.END).strip().startswith('Enter your text prompt'):
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.config(fg='black')
    
    def _on_text_focus_out(self, event):
        """Handle text widget focus out (add placeholder if empty)."""
        if not self.text_widget.get('1.0', tk.END).strip():
            placeholder = "Enter your text prompt here."
            self.text_widget.insert('1.0', placeholder)
            self.text_widget.config(fg='gray')
    
    def _on_text_change(self, event):
        """Handle text change."""
        text_content = self.text_widget.get('1.0', tk.END).strip()
        if text_content and not text_content.startswith('Enter your text prompt'):
            self.current_input_data = text_content
            if self.on_input_change:
                self.on_input_change("text", text_content)
    
    def _browse_image_file(self):
        """Open file dialog to select image."""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=file_types
        )
        
        if filename:
            if not os.path.exists(filename):
                messagebox.showerror("File Error", "Selected file not found.")
                return

            
            self.image_path_var.set(filename)
            self.current_input_data = filename
            self._update_image_preview(filename)
            
            if self.on_input_change:
                self.on_input_change("image", filename)
    
    def _browse_audio_file(self):
        """Open file dialog to select audio file."""
        file_types = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("M4A files", "*.m4a"),
            ("OGG files", "*.ogg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if filename:
            if not os.path.exists(filename):
                messagebox.showerror("File Error", "Selected file not found.")
                return
            
            self.audio_path_var.set(filename)
            self.current_input_data = filename
            self._update_audio_info(filename)
            
            if self.on_input_change:
                self.on_input_change("audio", filename)
    
    def _update_image_preview(self, image_path):
        """Update image preview."""
        try:
            # Open and resize image for preview
            with Image.open(image_path) as img:
                # Calculate size to fit in preview (max 10x10 as requested)
                img.thumbnail((50, 50), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Update preview label
                self.image_preview.config(image=photo, text="")
                self.image_preview.image = photo  # Keep reference
                
        except Exception as e:
            self.image_preview.config(
                image="",
                text=f"Preview error: {str(e)[:50]}..."
            )
            self.image_preview.image = None
    
    def _update_audio_info(self, audio_path):
        """Update audio file information display."""
        try:
            import os
            
            # Get basic file info
            file_size = os.path.getsize(audio_path)
            file_size_mb = file_size / (1024 * 1024)
            file_ext = os.path.splitext(audio_path)[1].upper()
            file_name = os.path.basename(audio_path)
            
            # Try to get audio duration if librosa is available
            duration_info = ""
            try:
                import librosa
                duration = librosa.get_duration(path=audio_path)
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_info = f"Duration: {minutes:02d}:{seconds:02d}\n"
            except Exception:
                duration_info = "Duration: Could not determine\n"
            
            info_text = f"""Selected: {file_name}
File format: {file_ext}
File size: {file_size_mb:.2f} MB
{duration_info}
"""
            
            self.audio_info.config(text=info_text)
            
        except Exception as e:
            self.audio_info.config(
                text=f"Audio file selected: {os.path.basename(audio_path)}\n\n"
                     f"Info error: {str(e)[:100]}...\n\n"
                     f"File ready for processing"
            )
        
        # Enable play button if audio file is loaded
        if hasattr(self, 'play_btn'):
            self.play_btn.config(state='normal')
    
    def _play_audio(self):
        """Play the selected audio file using pygame."""
        if not self.audio_path_var.get():
            messagebox.showwarning("No Audio", "Please select an audio file first.")
            return
        
        try:
            import pygame
            
            # Initialize pygame mixer if not already done
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Load and play the audio
            pygame.mixer.music.load(self.audio_path_var.get())
            pygame.mixer.music.play()
            
            # Enable stop button, disable play button temporarily
            if hasattr(self, 'stop_btn'):
                self.stop_btn.config(state='normal')
            
            self.play_btn.config(text="🔊 Playing...", state='disabled')
            
            # Check if playback finished (simple polling)
            self.after(100, self._check_playback_status)
            
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio file:\n{str(e)}")
    
    def _stop_audio(self):
        """Stop audio playback."""
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            
            # Reset button states
            if hasattr(self, 'play_btn'):
                self.play_btn.config(text="▶ Play Audio", state='normal')
            if hasattr(self, 'stop_btn'):
                self.stop_btn.config(state='disabled')
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not stop audio:\n{str(e)}")
    
    def _check_playback_status(self):
        """Check if audio playback is still active."""
        try:
            import pygame
            if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                # Playback finished
                if hasattr(self, 'play_btn'):
                    self.play_btn.config(text="▶ Play Audio", state='normal')
                if hasattr(self, 'stop_btn'):
                    self.stop_btn.config(state='disabled')
            else:
                # Still playing, check again
                self.after(100, self._check_playback_status)
        except Exception:
            # If there's an error, just reset the buttons
            if hasattr(self, 'play_btn'):
                self.play_btn.config(text="▶ Play Audio", state='normal')
            if hasattr(self, 'stop_btn'):
                self.stop_btn.config(state='disabled')
    
    def get_current_input(self) -> tuple[str, Any]:
        """
        Get current input type and data.
        
        Returns:
            Tuple of (input_type, input_data)
        """
        input_type = self.current_input_type.get()
        
        if input_type == "text":
            text_content = self.text_widget.get('1.0', tk.END).strip()
            if text_content and not text_content.startswith('Enter your text prompt'):
                return ("text", text_content)
            else:
                return ("text", None)
        
        elif input_type == "image":
            return ("image", self.current_input_data)
        
        elif input_type == "audio":
            return ("audio", self.current_input_data)
        
        return (input_type, None)
    
    def clear_input(self):
        """Clear current input."""
        input_type = self.current_input_type.get()
        
        if input_type == "text":
            self.text_widget.delete('1.0', tk.END)
            self._on_text_focus_out(None)  # Restore placeholder
        
        elif input_type == "image":
            self.image_path_var.set("")
            self.image_preview.config(image="", text="No image selected")
            self.image_preview.image = None
            self.current_input_data = None
        
        elif input_type == "audio":
            self.audio_path_var.set("")
            self.audio_info.config(text="No audio file selected\n\nSupported formats:\n• WAV (.wav)\n• MP3 (.mp3)\n• FLAC (.flac)\n• M4A (.m4a)\n• OGG (.ogg)")
            self.current_input_data = None


class ModelSelector(ttk.LabelFrame):
    """
    Custom widget for selecting AI models and displaying model information.
    
    This demonstrates:
    - Model information display
    - Dynamic content updates
    - Professional UI design
    """
    
    def __init__(self, parent, models_info: Dict[str, Dict], 
                 on_model_change: Callable = None, **kwargs):
        super().__init__(parent, text="Model Selection", **kwargs)
        
        self.models_info = models_info
        self.on_model_change = on_model_change
        self.current_model = tk.StringVar()
        
        self._setup_ui()
        self._bind_events()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Model selector
        selector_frame = ttk.Frame(self)
        selector_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(selector_frame, text="Select Model:").pack(side='left')
        
        # Store all model names for filtering
        self.all_model_names = [info['display_name'] for info in self.models_info.values()]
        self.current_filter = None  # Track current input type filter
        
        self.model_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.current_model,
            values=self.all_model_names,
            state="readonly",
            width=30
        )
        self.model_combo.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
        # Set default selection
        if self.all_model_names:
            self.model_combo.set(self.all_model_names[0])
        
        # Model info display
        self.info_frame = ttk.LabelFrame(self, text="Model Information")
        self.info_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.info_text = scrolledtext.ScrolledText(
            self.info_frame,
            height=8,
            wrap=tk.WORD,
            state='disabled',
            font=("TkDefaultFont", 9)
        )
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Update info display
        self._update_model_info()
    
    def _bind_events(self):
        """Bind event handlers."""
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
    
    def _on_model_change(self, event=None):
        """Handle model selection change."""
        self._update_model_info()
        
        if self.on_model_change:
            model_key = self.get_selected_model_key()
            self.on_model_change(model_key)
    
    def _update_model_info(self):
        """Update the model information display."""
        selected_display_name = self.current_model.get()
        
        # Find the model info by display name
        model_info = None
        for key, info in self.models_info.items():
            if info['display_name'] == selected_display_name:
                model_info = info
                break
        
        if not model_info:
            return
        
        # Format model information
        info_text = f"""Model: {model_info['name']}
Task: {model_info['task'].replace('-', ' ').title()}
Size: {model_info.get('model_size', 'Unknown')}
Author: {model_info.get('author', 'Unknown')}
License: {model_info.get('license', 'Unknown')}

Description:
{model_info['description']}

Input Type: {model_info['input_type'].title()}
Output Type: {model_info['output_type'].replace('_', ' ').title()}

Example Usage:
{model_info.get('example_usage', 'No example available')}

Requirements:
{', '.join(model_info.get('requirements', ['None']))}

Hugging Face URL:
{model_info.get('huggingface_url', 'Not available')}"""
        
        # Update the text widget
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', info_text)
        self.info_text.config(state='disabled')
    
    def get_selected_model_key(self) -> Optional[str]:
        """
        Get the key of the currently selected model.
        
        Returns:
            Model key or None if no selection
        """
        selected_display_name = self.current_model.get()
        
        for key, info in self.models_info.items():
            if info['display_name'] == selected_display_name:
                return key
        
        return None
    
    def set_selected_model(self, model_key: str):
        """
        Set the selected model by key.
        
        Args:
            model_key: Key of the model to select
        """
        if model_key in self.models_info:
            display_name = self.models_info[model_key]['display_name']
            self.current_model.set(display_name)
            self._update_model_info()
    
    def filter_models_by_input_type(self, input_type: str):
        """
        Filter available models based on input type.
        
        Args:
            input_type: Type of input ('text', 'image', 'audio')
        """
        self.current_filter = input_type
        
        # Find models that match the input type
        compatible_models = []
        compatible_keys = []
        
        for key, info in self.models_info.items():
            if info['input_type'] == input_type:
                compatible_models.append(info['display_name'])
                compatible_keys.append(key)
        
        # Update frame title to show filtering
        input_type_display = input_type.title()
        filter_count = len(compatible_models)
        total_count = len(self.all_model_names)
        
        if filter_count < total_count:
            self.info_frame.config(text=f"Model Information - {input_type_display} Models ({filter_count}/{total_count})")
        else:
            self.info_frame.config(text="Model Information")
        
        # Update combobox values
        if compatible_models:
            self.model_combo['values'] = compatible_models
            
            # Auto-select the best model for this input type
            current_selection = self.current_model.get()
            if current_selection not in compatible_models:
                # Select the first compatible model
                best_model = self._get_best_model_for_input_type(input_type, compatible_keys)
                if best_model:
                    self.current_model.set(self.models_info[best_model]['display_name'])
                else:
                    self.current_model.set(compatible_models[0])
                self._update_model_info()
                
                # Trigger model change callback if available
                if self.on_model_change:
                    selected_key = self.get_selected_model_key()
                    if selected_key:
                        self.on_model_change(selected_key)
        else:
            # No compatible models found - show all models
            self.model_combo['values'] = self.all_model_names
    
    def _get_best_model_for_input_type(self, input_type: str, compatible_keys: list) -> str:
        """
        Get the best model for a specific input type.
        
        Args:
            input_type: Type of input ('text', 'image', 'audio')
            compatible_keys: List of compatible model keys
            
        Returns:
            Best model key for the input type
        """
        # Define preferences for each input type
        preferences = {
            'text': ['sentiment_classifier', 'text_to_image'],  # Prefer sentiment analysis, then text-to-image for text
            'image': ['image_classifier'],  # Only image classifier for images
            'audio': ['speech_to_text']     # Only speech-to-text for audio
        }
        
        # Find the preferred model
        for preferred in preferences.get(input_type, []):
            if preferred in compatible_keys:
                return preferred
        
        # Return first compatible if no preference found
        return compatible_keys[0] if compatible_keys else None
    
    def reset_model_filter(self):
        """Reset model filter to show all models."""
        self.current_filter = None
        self.model_combo['values'] = self.all_model_names
        self.info_frame.config(text="Model Information")
    
    def get_current_filter(self) -> str:
        """Get the current input type filter."""
        return self.current_filter


class ProcessingControls(ttk.Frame):
    """
    Custom widget for model processing controls.
    
    This demonstrates:
    - Control panel design
    - Status indicator integration
    - User feedback mechanisms
    """
    
    def __init__(self, parent, on_run: Callable = None, 
                 on_clear: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.on_run = on_run
        self.on_clear = on_clear
        self.is_processing = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the control interface."""
        # Main buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(side='left', padx=5)
        
        self.run_button = ttk.Button(
            button_frame,
            text="▶ Run Model",
            command=self._on_run_click,
            style="Accent.TButton"
        )
        self.run_button.pack(side='left', padx=2)
        
        self.clear_button = ttk.Button(
            button_frame,
            text="🗑 Clear",
            command=self._on_clear_click
        )
        self.clear_button.pack(side='left', padx=2)
        
        # Status frame
        status_frame = ttk.Frame(self)
        status_frame.pack(side='right', padx=5)
        
        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            font=("TkDefaultFont", 9)
        )
        self.status_label.pack(side='right')
        
        # Progress bar (hidden by default)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            mode='indeterminate',
            length=100
        )
        # Don't pack initially - will be shown during processing
    
    def _on_run_click(self):
        """Handle run button click."""
        if self.on_run and not self.is_processing:
            # Set processing state immediately
            self.set_processing_state(True, status_text="Processing...")
            
        # Run the on_run callback in a separate thread
        threading.Thread(target=self._run_task_thread, daemon=True).start()
    
    def _run_task_thread(self):
        """Wrapper to execute on_run and handle completion."""
        try:
            self.on_run()  # Run the long-running task
        except Exception as e:
            # Show error in status label
            self.status_label.config(text=f"Error: {str(e)}")
        finally:
            # Restore UI after processing
            self.set_processing_state(False)
    
    def _on_clear_click(self):
        """Handle clear button click."""
        if self.on_clear and not self.is_processing:
            self.on_clear()
    
    def set_processing_state(self, is_processing: bool, status_text: str = None):
        """
        Update the processing state and UI.
        
        Args:
            is_processing: Whether processing is active
            status_text: Optional status message
        """
        self.is_processing = is_processing
        
        if is_processing:
            self.run_button.config(state='disabled', text=" Processing...")
            self.clear_button.config(state='disabled')
            self.status_label.pack_forget()
            self.progress_bar.pack(side='right', padx=(0, 5))
            self.progress_bar.start()
            
            if status_text:
                self.status_label.config(text=status_text)
        else:
            self.run_button.config(state='normal', text=" Run Model")
            self.clear_button.config(state='normal')
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self.status_label.pack(side='right')
            
            if status_text:
                self.status_label.config(text=status_text)
            else:
                self.status_label.config(text="Ready")


# Example usage
if __name__ == "__main__":
    # Test the custom widgets
    root = tk.Tk()
    root.title("Widget Test")
    root.geometry("800x600")
    
    # Test input selector
    def on_input_change(input_type, data):
        print(f"Input changed: {input_type}, {data}")
    
    input_selector = InputSelector(root, on_input_change=on_input_change)
    input_selector.pack(fill='both', expand=True, padx=10, pady=5)
    
    # Test model selector
    mock_models = {
        'test_model': {
            'display_name': 'Test Model',
            'name': 'test/model',
            'task': 'test-task',
            'description': 'A test model for demonstration',
            'input_type': 'text',
            'output_type': 'text'
        }
    }
    
    def on_model_change(model_key):
        print(f"Model changed: {model_key}")
    
    model_selector = ModelSelector(root, mock_models, on_model_change=on_model_change)
    # model_selector.pack(fill='both', expand=True, padx=10, pady=5)
    
    root.mainloop()
    
