# CLAUDE.md - Project NEAT

## Always Do This
- Read @PLAN_MAIN and @TECHNICALd.md OR keep them in context.
- Review current Project NEAT status, determine where we left off if not told, and determine the best path forward to complete the given task and align with the theory in the @TECHNICALd.md
- Your main goal is to complete this project, so think holisticly and deeply about how exactly would be best to integrate all components and create a truly breakthrough algorithm. Be honest and accurate, do not ever sandbag tests or take the easy route. Make effecive and as simple as possible chages, but do not shy away from challges, simply adjsut the steps manually in @PLAN_MD and then continue with the further broken down task if needed. Don't get overwhelemed, stay calm, just code and think hard.
- Use pytest inteigently to verify all components and processes that feasibly can be, creating comprehesive platfrom agnostic robust and efficent tests that do not sandbag or mis use mocks. Solid, professional and smart tests.
- This project will be run on mac m3 apple silicon and windows 11 x86 (read metal_docs.md for details on how to use pytorch with apples metal framework)
- Update progress regulary and anything you learn about the project to this document that you deem very important to remember forever.


## Build & Test Commands
- Run all tests: `python -m pytest tests/`
- Run a single test: `python -m pytest tests/test_components.py::TestName::test_method_name`
- Train model: `python main.py --mode train --use_titans_memory --use_transformer2_adaptation --use_mvot_processor`
- Evaluate model: `python main.py --mode eval --model_path ./outputs/best_model`
- Profile components: `python main.py --mode profile`

## Code Style Guidelines
- Use type hints (from typing import Dict, List, Optional, Union, Any)
- Add docstrings for all modules, classes, and methods
- Imports order: standard library, third-party, local modules
- Use snake_case for variables/methods, CamelCase for classes
- Exception handling: catch specific exceptions
- Python 3.8+ features are allowed
- Class structure: __init__ first, then public methods, then private methods (_method_name)
- Follow PyTorch conventions for tensor operations
- Use absolute imports (from src.utils... not from ..utils...)
- Return dict for model outputs with keys like "logits", "hidden_states"