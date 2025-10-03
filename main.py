"""
Main entry point for the HIT137 Assignment 3 AI Model Integration Application.

This application demonstrates advanced OOP concepts while providing a user-friendly
interface for interacting with Hugging Face AI models.

Author: [Your Team Name]
Date: 2025
Course: HIT137
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.app_window import AIModelApp


def main():
    """
    Initialize and run the AI Model Integration Application.
    
    Creates the main Tkinter window and starts the application event loop.
    Handles any startup errors gracefully.
    """
    try:
        # Create the main application window
        root = tk.Tk()
        
        
        # Initialize the application
        app = AIModelApp(root)
        
        # Start the main event loop
        root.mainloop()
        
    except ImportError as e:
        messagebox.showerror(
            "Import Error", 
            f"Missing required dependencies: {e}\n\n"
            "Please install requirements:\npip install -r requirements.txt"
        )
    except Exception as e:
        messagebox.showerror(
            "Application Error", 
            f"An unexpected error occurred: {e}\n\n"
            "Please check the console for more details."
        )
        print(f"Error details: {e}")


if __name__ == "__main__":
    main()