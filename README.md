# HIT137 Group Assignment 3 · AI Model Integration Platform

## 1. Project Overview
The application showcases how disciplined object-oriented design keeps an applied AI solution maintainable. A Tkinter desktop client orchestrates multiple Hugging Face models (sentiment analysis, image classification, speech-to-text, text-to-image) while demonstrating patterns such as factories, mixins, polymorphism, and threaded execution for a responsive user experience.

## 2. Key Capabilities
- **Unified workspace** – switch between text, image, audio, and prompt-driven generation from a single control panel.
- **Model catalogue** – view preparation tips, interpretation guidance, and operational constraints for each integrated model.
- **Rich visual feedback** – result cards, sortable tables, and contextual annotations clarify what the models produced and why.
- **Robust architecture** – separation between GUI, model factory, and Hugging Face wrappers keeps the codebase modular and testable.

## 3. Architecture Snapshot
```
HIT137-groupX-assignment3/
├── main.py                     # Application entry point (logging + Tk bootstrap)
├── gui/
│   ├── app_window.py           # Main window + layout orchestration
│   ├── widgets.py              # Input selector, model selector, control strip
│   ├── ui_helpers.py           # Results renderer + OOP explainer
│   └── theme.py                # Centralised visual styling
├── models/
│   ├── model_factory.py        # Factory pattern with caching
│   ├── hf_integration.py       # Hugging Face wrapper classes
│   └── model_info.py           # Metadata + usage guidance
├── oop_examples/               # Reusable OOP demonstrations
│   ├── base_classes.py         # Interfaces, mixins, multiple inheritance
│   └── decorators.py           # Timing, logging, retry decorators
├── tests/
│   └── simple_tests.py         # Lightweight regression checks
└── logging_config.py           # Structured logging configuration
```

## 4. Technology Stack
- **Python 3.10+**
- **Tkinter / ttk themed widgets** for the desktop UI
- **Hugging Face Transformers & Diffusers** for model execution
- **PyTorch** backend with CPU/MPS fallbacks
- **Pillow, librosa, soundfile** for media handling

## 5. Object-Oriented Patterns in Focus
| Pattern / Concept       | Implementation Highlights                                                |
|-------------------------|----------------------------------------------------------------------------|
| Multiple inheritance     | `MultiInheritanceModelWrapper` composes base behaviours and mixins        |
| Encapsulation            | Private `_initialize_model` routines hide Hugging Face boilerplate        |
| Polymorphism             | Every wrapper satisfies `BaseModelInterface.run()` for uniform control    |
| Method overriding        | Input validation and `run()` signatures tailored per modality             |
| Factory pattern          | `ModelFactory.create_model()` selects and caches wrappers seamlessly      |
| Decorator stacking       | `@timeit`, `@log_exceptions`, `@retry_on_failure` enhance model methods   |
| Event-driven updates     | GUI reacts to input/model changes to keep controls and notes in sync      |

## 6. Refined User Experience
- Light, professional theme with clear typography and contrast for inputs and text areas.
- Accessible placeholders and status messaging to guide data entry.
- Commentary panes summarise how to interpret results, when to be cautious, and what to revisit.
- Model information auto-refreshes with team-oriented operational notes.

## 7. Getting Started
1. **Clone the repository**
   ```bash
   git clone https://github.com/mbishnoi29786/HIT137-groupX-assignment3.git
   cd HIT137-groupX-assignment3
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the application**
   ```bash
   python main.py
   ```

> **Note:** On macOS, the application constrains BLAS/OpenMP threading to avoid the Accelerate SIGBUS issue and will automatically fall back to CPU or MPS where appropriate.

## 8. Using the Application
1. Choose an **input type** (text, image, audio, or prompt for generation).
2. Provide the relevant **input data** (free-text, file chooser, etc.).
3. Select a **model** from the filtered list.  The model information tab updates instantly.
4. Press **Run Model** to execute inference; background threads keep the interface responsive.
5. Review **Results**, inspect the **OOP Concepts** reference, or read the **Model Details** tab for context.
6. Export structured output to JSON for downstream analysis if required.

## 9. Integrated Models (Summary)
| Model Key              | Hugging Face ID                                  | Primary Use Case                                  |
|------------------------|--------------------------------------------------|---------------------------------------------------|
| sentiment_classifier   | distilbert-base-uncased-finetuned-sst-2-english  | Rapid binary sentiment scoring                    |
| image_classifier       | google/vit-base-patch16-224                      | General-purpose image categorisation              |
| speech_to_text         | openai/whisper-tiny                              | Lightweight audio transcription                   |
| text_to_image          | stabilityai/stable-diffusion-2                   | Concept art and visual ideation from text prompts |

Each entry in the UI includes preparation checklists, interpretation guidance, operational constraints, external references, and dependency notes.

## 10. Logging & Diagnostics
Structured logging is initialised via `logging_config.py` before any module import:
- Console stream at INFO for operator awareness
- Rotating file handler (`app.log`) capturing DEBUG detail
- Background inference threads log status transitions to support troubleshooting

## 11. Testing
Run the regression tests whenever dependencies or model wrappers change:
```bash
python -m pytest tests/
```
Additional unit or integration tests can be layered onto `tests/` as new features are introduced.

## 12. Team Members
- **Manish** · Student ID S393232
- **Oshan Thapa Chhetri** · Student ID S395087
- **Iresh Maharjan** · Student ID S396815
- **Karma Sonam Manandhar** · Student ID S396680

## 13. Contribution Workflow
- Branch-per-feature with descriptive commit messages
- Pull requests reviewed by peers before merging
- Linting and targeted tests prior to submission
- Documentation and UI copy kept in step with feature changes

## 14. License & Academic Context
This repository forms part of the HIT137 coursework (Sydney Group 24). The project is intended for educational use and demonstrates best practices for collaborative software delivery within an academic setting.
