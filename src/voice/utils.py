"""Utility functions for voice processing."""

import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for TTS streaming.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []

    # Simple sentence splitting on common terminators
    # Handles: . ! ? with optional quotes/parentheses
    sentences = re.split(r'([.!?]+[\"\'\)]*\s+)', text)

    # Recombine split parts
    result = []
    current = ""

    for i, part in enumerate(sentences):
        current += part

        # If this is a terminator (odd index) or last part
        if i % 2 == 1 or i == len(sentences) - 1:
            sentence = current.strip()
            if sentence:
                result.append(sentence)
            current = ""

    return result if result else [text.strip()]
