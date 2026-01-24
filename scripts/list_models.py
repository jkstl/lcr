#!/usr/bin/env python3
"""List all available Ollama models on this system."""

import asyncio
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_ollama_models():
    """Get list of installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        models = []

        for line in lines:
            if line.strip():
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(model_name)

        return models
    except subprocess.CalledProcessError as e:
        print(f"Error running ollama: {e}")
        return []
    except FileNotFoundError:
        print("Ollama not found. Is it installed?")
        return []


def get_model_info(model_name):
    """Get detailed info about a model."""
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except:
        return None


def main():
    print("=" * 80)
    print("AVAILABLE OLLAMA MODELS")
    print("=" * 80)
    print()

    models = get_ollama_models()

    if not models:
        print("No models found!")
        return

    # Categorize models
    qwen_models = [m for m in models if 'qwen' in m.lower()]
    llama_models = [m for m in models if 'llama' in m.lower()]
    embed_models = [m for m in models if 'embed' in m.lower() or 'nomic' in m.lower()]
    other_models = [m for m in models if m not in qwen_models + llama_models + embed_models]

    print("Main LLM Models (for conversation):")
    print("-" * 40)
    for model in qwen_models + llama_models + other_models:
        if 'embed' not in model.lower():
            print(f"  • {model}")

    print()
    print("Embedding Models (for vector search):")
    print("-" * 40)
    for model in embed_models:
        print(f"  • {model}")

    print()
    print("=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    print()

    from src.config import settings

    print(f"Main Model:            {settings.main_model}")
    print(f"Observer Utility:      {settings.observer_utility_model}")
    print(f"Observer Extraction:   {settings.observer_extraction_model}")
    print(f"Embedding Model:       {settings.embedding_model}")

    print()
    print("=" * 80)
    print("HOW TO CHANGE MODELS")
    print("=" * 80)
    print()
    print("Option 1: Edit .env file (create from .env.example if needed)")
    print("  MAIN_MODEL=qwen3:4b")
    print("  OBSERVER_UTILITY_MODEL=qwen3:0.6b")
    print("  OBSERVER_EXTRACTION_MODEL=nuextract:numind/NuExtract-2.0-2B")
    print("  EMBEDDING_MODEL=nomic-embed-text")
    print()
    print("Option 2: Set environment variables")
    print("  export MAIN_MODEL=qwen3:4b")
    print("  export OBSERVER_UTILITY_MODEL=qwen3:1.7b")
    print("  export OBSERVER_EXTRACTION_MODEL=nuextract:numind/NuExtract-2.0-2B")
    print()
    print("Option 3: Use the model testing tool")
    print("  python scripts/model_tester.py")
    print()


if __name__ == "__main__":
    main()
