"""Voice module for TTS and STT functionality."""

from .tts import TTSEngine, VoiceConfig
from .utils import split_into_sentences

__all__ = ["TTSEngine", "VoiceConfig", "split_into_sentences"]
