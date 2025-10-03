"""
Setup and run script for the HIT137 Assignment 3 AI Model Integration Application.

This script provides utilities for setting up the environment, running tests,
and launching the application with proper error handling.

Run this script to:
1. Check system requirements
2. Install dependencies
3. Run comprehensive tests
4. Launch the application

Usage:
    python setup_and_run.py [--test-only] [--no-install] [--help]
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"Python {sys.version.split()[0]} - OK")
    return True


def check_system_requirements():
    """Check system requirements and provide recommendations."""
    print("\nChecking system requirements...")
    
    # Check available memory (basic check)
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"   RAM: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        
        if total_gb < 4:
            print("   Warning: Less than 4GB RAM - some models may be slow")
        elif total_gb >= 8:
            print("   Sufficient RAM for all AI models")
        else:
            print("   Adequate RAM (4-8GB) - models will work")
            
    except ImportError:
        print("   RAM check unavailable (psutil not installed)")
        print("   Recommend: 4GB+ RAM for optimal performance")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"   Disk space: {free_gb:.1f}GB available")
        
        if free_gb < 10:
            print("   Warning: Less than 10GB free - models may fail to download")
            return False
        else:
            print("   Sufficient disk space for AI models")
            
    except Exception:
        print("   Disk space check unavailable")
        print("   Recommend: 10GB+ free space for model downloads")
    
    # Platform-specific notes
    import platform
    system = platform.system()
    print(f"   Platform: {system} {platform.release()}")
    
    if system == "Windows":
        print("   Windows detected - all features supported")
    elif system == "Darwin":  # macOS
        print("   macOS detected - all features supported")
    elif system == "Linux":
        print("   Linux detected - all features supported")
        print("   Note: Audio playback requires ALSA/PulseAudio")

    return True


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n Checking dependencies...")
    
    required_packages = [
        ('tkinter', 'tkinter'),
        ('PIL', 'PIL.Image'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('diffusers', 'diffusers'),
        ('librosa', 'librosa'),
        ('soundfile', 'soundfile'),
        ('pygame', 'pygame'),
        ('requests', 'requests'),
        ('numpy', 'numpy')
    ]
    
    missing_packages = []
    
    for display_name, import_name in required_packages:
        try:
            if import_name == 'PIL.Image':
                from PIL import Image
            else:
                __import__(import_name)

            print(f" {display_name} - Available")
        except ImportError:
            print(f" {display_name} - Missing")
            missing_packages.append(display_name)
    
    return missing_packages


def install_dependencies():
    """Install required dependencies."""
    print("\n Installing dependencies...")
    print("   This may take several minutes for first-time installation...")
    
    try:
        # Update pip first
        print("   Updating pip...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
        ], capture_output=True, text=True, timeout=60)
        
        # Install from requirements.txt with verbose output
        print("Installing AI model dependencies...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--upgrade'
        ], capture_output=False, text=True, timeout=600)  # 10 minutes timeout
        
        if result.returncode == 0:
            print("\n Dependencies installed successfully")
            print(" All AI models and audio processing ready!")
            return True
        else:
            print(f"\n Installation failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n Installation timed out (>10 minutes)")
        print("   Try installing manually: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n Installation error: {e}")
        return False


def run_tests():
    """Run the comprehensive test suite."""
    print("\n Running comprehensive tests...")

    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import and run tests
        from tests.simple_tests import generate_test_report
        
        success = generate_test_report()
        
        if success:
            print("\n All tests passed! Application is ready to run.")
            return True
        else:
            print("\n Some tests failed. Please check the output above.")
            return False
            
    except ImportError as e:
        print(f" Could not import test modules: {e}")
        print("   This might be due to missing dependencies.")
        return False
    except Exception as e:
        print(f" Test execution error: {e}")
        return False


def launch_application():
    """Launch the main application."""
    print("\n Launching AI Model Integration Application...")
    
    try:
        # Import and run the main application
        from main import main
        
        print("✅ Application started successfully!")
        print("   Use the GUI to interact with AI models.")
        print("   Close the window to exit.")
        
        main()
        
    except ImportError as e:
        print(f" Could not import main application: {e}")
        return False
    except Exception as e:
        print(f" Application launch error: {e}")
        return False
    
    return True


def print_project_info():
    """Print project information and structure."""
    print("=" * 70)
    print(" HIT137 ASSIGNMENT 3 - AI MODEL INTEGRATION APPLICATION")
    print("=" * 70)
    print()
    print(" Project Overview:")
    print("   • Demonstrates advanced OOP concepts with AI model integration")
    print("   • 4 AI model types: text, image, audio, and text-to-image")
    print("   • Professional Tkinter GUI with intelligent model filtering")
    print("   • Audio playback, speech-to-text (no ffmpeg required)")
    print("   • Scrollable UI, export functionality, comprehensive testing")
    print()
    print(" OOP Concepts Demonstrated:")
    print("   ✓ Multiple Inheritance")
    print("   ✓ Encapsulation")
    print("   ✓ Polymorphism")
    print("   ✓ Method Overriding")
    print("   ✓ Multiple Decorators")
    print("   ✓ Factory Pattern")
    print("   ✓ Observer Pattern")
    print()
    print(" AI Models Integrated:")
    print("   • Sentiment Analysis: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)")
    print("   • Image Classification: Vision Transformer (google/vit-base-patch16-224)")
    print("   • Speech-to-Text: Whisper Tiny (openai/whisper-tiny)")
    print("   • Text-to-Image: Stable Diffusion 2.0 (stabilityai/stable-diffusion-2)")
    print()
    print(" Key Features:")
    print("   ✓ Intelligent model filtering based on input type")
    print("   ✓ Audio playback with play/stop controls")
    print("   ✓ Speech-to-text without ffmpeg dependency")
    print("   ✓ Export results functionality")
    print("   ✓ Scrollable UI with fixed heights")
    print("   ✓ Auto-clear results on new runs")
    print("   ✓ 10x10 pixel image previews")
    print("   ✓ Threaded processing with progress indicators")
    print()
    print(" Project Structure:")
    
    structure = """   HIT137-Assignment3/
   ├── main.py                      # Application entry point
   ├── setup_and_run.py            # Complete setup and launcher
   ├── requirements.txt             # Python dependencies (updated)
   ├── README.md                    # Project documentation
   ├── github_link.txt              # Repository URL
   ├── FIXES_IMPLEMENTED.md         # Latest fixes documentation
   ├── UI_FIXES_SUMMARY.md          # UI improvements summary
   ├── gui/                         # GUI components
   │   ├── app_window.py            # Main application window (enhanced)
   │   ├── widgets.py               # Custom UI widgets (with audio playback)
   │   └── ui_helpers.py            # UI utility functions (scrollable, export)
   ├── models/                      # AI model integration
   │   ├── hf_integration.py        # 4 model types (no ffmpeg required)
   │   ├── model_info.py            # Model metadata and usage guides
   │   └── model_factory.py         # Model creation factory
   ├── oop_examples/                # OOP demonstrations
   │   ├── base_classes.py          # Base classes and interfaces
   │   └── decorators.py            # Custom decorators
   ├── tests/                       # Test files
   │   ├── simple_tests.py          # Comprehensive test suite
   │   ├── test_fixes.py            # Audio/ffmpeg fixes tests
   │   ├── test_ui_fixes.py         # UI improvements tests
   │   └── test_all_models.py       # All 4 models functionality tests"""
    
    print(structure)
    print()


def main():
    """Main setup and run function."""
    parser = argparse.ArgumentParser(
        description="Setup and run the HIT137 Assignment 3 AI Model Integration Application"
    )
    parser.add_argument(
        '--test-only', 
        action='store_true', 
        help='Run tests only, do not launch application'
    )
    parser.add_argument(
        '--no-install', 
        action='store_true', 
        help='Skip dependency installation'
    )
    parser.add_argument(
        '--info-only', 
        action='store_true', 
        help='Show project information only'
    )
    
    args = parser.parse_args()
    
    # Print project information
    print_project_info()
    
    if args.info_only:
        return
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n  System requirements not met. Continue at your own risk.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check and install dependencies
    if not args.no_install:
        missing_deps = check_dependencies()
        
        if missing_deps:
            print(f"\n  Missing dependencies: {', '.join(missing_deps)}")
            if not install_dependencies():
                print("\n Setup failed. Please install dependencies manually:")
                print("   pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("\n All dependencies are available")
    
    # Run tests
    if not run_tests():
        print("\n Some tests failed, but you can still try running the application.")
        
        response = input("\nDo you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Launch application (unless test-only mode)
    if not args.test_only:
        print("\n" + "=" * 70)
        success = launch_application()
        
        if success:
            print("\n🎉 Thank you for using the AI Model Integration Application!")
        else:
            print("\n Application failed to launch. Please check the error messages above.")
            sys.exit(1)
    else:
        print("\n Test-only mode completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Setup interrupted by user.")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)