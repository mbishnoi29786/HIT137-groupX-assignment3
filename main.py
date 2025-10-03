"""
Main entry point for the HIT137 Assignment 3 AI Model Integration Application.

This application demonstrates advanced OOP concepts while providing a user-friendly
interface for interacting with Hugging Face AI models.
It includes robust error handling and clear user feedback mechanisms.
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import logging
import platform

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logging_config import initialize_logging

# Configure logging before importing app modules that define loggers
initialize_logging()

from gui.app_window import AIModelApp


def main():
    """
    Initialize and run the AI Model Integration Application.
    
    Creates the main Tkinter window and starts the application event loop.
    Handles any startup errors gracefully.
    """
    logger = logging.getLogger(__name__)

    try:
        logger.info(
            "Application starting on %s %s | Tk version=%s",
            platform.system(),
            platform.release(),
            tk.TkVersion,
        )
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
        logger.exception("Failed to import dependency")
    except Exception as e:
        messagebox.showerror(
            "Application Error", 
            f"An unexpected error occurred: {e}\n\n"
            "Please check the console for more details."
        )
        print(f"Error details: {e}")
        logger.exception("Unhandled exception during application execution")


if __name__ == "__main__":
    main()
