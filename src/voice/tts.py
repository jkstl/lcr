"""Text-to-Speech engine using Kokoro TTS."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import urllib.request

import numpy as np
import sounddevice as sd

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceConfig:
    """Configuration for TTS voice."""

    voice: str = "af_heart"  # Default female voice (highest quality)
    speed: float = 1.0
    enabled: bool = False

    # Available Kokoro female voices (American English)
    # Source: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
    FEMALE_VOICES = [
        "af_heart",    # A grade - Highest quality
        "af_bella",    # A- grade - Clear, professional
        "af_nicole",   # B- grade - Expressive
        "af_sarah",    # C+ grade - Natural
        "af_aoede",    # C+ grade
        "af_kore",     # C+ grade
        "af_nova",     # C grade
        "af_alloy",    # C grade
        "af_sky",      # C- grade - Bright
        "af_jessica",  # D grade
        "af_river",    # D grade
    ]


class TTSEngine:
    """
    Kokoro TTS engine for natural-sounding speech synthesis.

    Features:
    - High-quality female voices
    - Fast inference (~210x real-time on GPU, 3-11x on CPU)
    - Lightweight (82M parameters, ~2-3GB VRAM)
    - Sentence-by-sentence streaming
    """

    # Model file URLs
    MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self._model = None
        self._initialized = False
        self._model_dir = Path("./data/models/kokoro")

    def _download_models(self):
        """Download Kokoro model files if they don't exist."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        model_path = self._model_dir / "kokoro-v1.0.onnx"
        voices_path = self._model_dir / "voices-v1.0.bin"

        # Download model if missing
        if not model_path.exists():
            LOGGER.info("Downloading Kokoro TTS model (~100MB)...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                LOGGER.info(f"✓ Model downloaded to {model_path}")
            except Exception as e:
                LOGGER.error(f"Failed to download model: {e}")
                raise

        # Download voices if missing
        if not voices_path.exists():
            LOGGER.info("Downloading Kokoro voice embeddings...")
            try:
                urllib.request.urlretrieve(self.VOICES_URL, voices_path)
                LOGGER.info(f"✓ Voices downloaded to {voices_path}")
            except Exception as e:
                LOGGER.error(f"Failed to download voices: {e}")
                raise

        return str(model_path), str(voices_path)

    def _lazy_load(self):
        """Lazy load Kokoro model on first use to save startup time."""
        if self._initialized:
            return

        try:
            # Import here to avoid loading if TTS is disabled
            from kokoro_onnx import Kokoro

            LOGGER.info(f"Loading Kokoro TTS model with voice: {self.config.voice}")

            # Download models if needed
            model_path, voices_path = self._download_models()

            # Initialize Kokoro
            self._model = Kokoro(
                model_path=model_path,
                voices_path=voices_path,
            )

            self._initialized = True
            LOGGER.info("✓ Kokoro TTS initialized successfully")

        except ImportError as e:
            LOGGER.error(f"Failed to import Kokoro TTS: {e}")
            LOGGER.error("Install with: pip install kokoro-onnx soundfile")
            raise
        except Exception as e:
            LOGGER.error(f"Failed to initialize Kokoro TTS: {e}")
            raise

    def _normalize_audio_output(self, audio_output) -> Optional[tuple[np.ndarray, int]]:
        """Normalize Kokoro output to (audio, sample_rate)."""
        if audio_output is None:
            return None

        sample_rate = 24000
        audio = audio_output

        if isinstance(audio_output, (tuple, list)) and len(audio_output) == 2:
            candidate_audio, candidate_rate = audio_output
            if isinstance(candidate_rate, (int, float)):
                audio = candidate_audio
                sample_rate = int(candidate_rate)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 0:
            return None

        return audio, sample_rate

    async def synthesize(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """
        Synthesize text to audio.

        Args:
            text: Text to convert to speech

        Returns:
            Audio array and sample rate or None if disabled/failed
        """
        if not self.config.enabled:
            return None

        if not text or not text.strip():
            return None

        try:
            # Lazy load model
            if not self._initialized:
                await asyncio.to_thread(self._lazy_load)

            # Generate audio in thread pool (blocking operation)
            # kokoro_onnx API: create(text, voice, speed, lang)
            audio = await asyncio.to_thread(
                self._model.create,
                text,
                self.config.voice,
                self.config.speed,
                "en-us",
            )

            return self._normalize_audio_output(audio)

        except Exception as e:
            LOGGER.error(f"TTS synthesis failed: {e}")
            return None

    async def speak(self, text: str):
        """
        Synthesize and play audio.

        Args:
            text: Text to speak
        """
        result = await self.synthesize(text)

        if result is None:
            return

        try:
            audio, sample_rate = result
            # Play audio using sounddevice
            await asyncio.to_thread(
                sd.play,
                audio,
                samplerate=sample_rate,
                blocking=True,
            )
        except Exception as e:
            LOGGER.error(f"Audio playback failed: {e}")

    async def speak_streaming(self, sentences: list[str]):
        """
        Speak multiple sentences with streaming generation.

        Starts playing the first sentence while synthesizing the rest in parallel.
        This prevents audio gaps between sentences.

        Args:
            sentences: List of sentences to speak
        """
        if not self.config.enabled or not sentences:
            return

        try:
            # Generate and play first sentence immediately
            if sentences:
                first_result = await self.synthesize(sentences[0])
                if first_result is not None:
                    first_audio, first_sample_rate = first_result
                    # Start playing first sentence
                    play_task = asyncio.create_task(
                        asyncio.to_thread(
                            sd.play,
                            first_audio,
                            samplerate=first_sample_rate,
                            blocking=True,
                        )
                    )

            # Synthesize all remaining sentences in parallel while first plays
            remaining_audios = []
            if len(sentences) > 1:
                synthesis_tasks = [
                    self.synthesize(sentence) for sentence in sentences[1:]
                ]
                results = await asyncio.gather(*synthesis_tasks, return_exceptions=True)

                # Filter out failures and exceptions
                for result in results:
                    if isinstance(result, Exception):
                        LOGGER.error(f"TTS synthesis failed: {result}")
                    elif result is not None:
                        remaining_audios.append(result)

            # Wait for first sentence to finish
            if 'play_task' in locals():
                await play_task

            # Play remaining sentences (already synthesized, no gaps)
            for audio, sample_rate in remaining_audios:
                try:
                    await asyncio.to_thread(
                        sd.play,
                        audio,
                        samplerate=sample_rate,
                        blocking=True,
                    )
                except Exception as e:
                    LOGGER.error(f"Audio playback failed: {e}")
                    break
        except asyncio.CancelledError:
            sd.stop()
            raise

    def set_voice(self, voice: str) -> bool:
        """
        Change voice and reload model.

        Args:
            voice: Voice name (e.g., "af_sarah", "af_bella")

        Returns:
            True if voice changed successfully
        """
        if voice not in VoiceConfig.FEMALE_VOICES:
            LOGGER.warning(f"Invalid voice: {voice}. Available: {VoiceConfig.FEMALE_VOICES}")
            return False

        self.config.voice = voice
        self._initialized = False  # Force reload on next use
        LOGGER.info(f"Voice changed to: {voice}")
        return True

    def set_speed(self, speed: float):
        """Set speech speed multiplier (0.5 = slow, 2.0 = fast)."""
        self.config.speed = max(0.5, min(2.0, speed))
        LOGGER.info(f"Speech speed set to: {self.config.speed}x")

    def enable(self):
        """Enable TTS output."""
        self.config.enabled = True
        LOGGER.info("TTS enabled")

    def disable(self):
        """Disable TTS output."""
        self.config.enabled = False
        LOGGER.info("TTS disabled")

    def toggle(self) -> bool:
        """Toggle TTS on/off. Returns new state."""
        self.config.enabled = not self.config.enabled
        LOGGER.info(f"TTS {'enabled' if self.config.enabled else 'disabled'}")
        return self.config.enabled
