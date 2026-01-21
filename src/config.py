try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover
    from pydantic import BaseSettings


class Settings(BaseSettings):
    # LLM
    ollama_host: str = "http://localhost:11434"
    main_model: str = "qwen3:14b"
    observer_model: str = "qwen3:1.7b"
    embedding_model: str = "nomic-embed-text"

    # Databases
    lancedb_path: str = "./data/lancedb"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_graph_id: str = "lcr_memories"
    redis_host: str = "localhost"
    redis_port: int = 6380

    # Memory
    max_context_tokens: int = 3000
    sliding_window_tokens: int = 2000
    vector_search_top_k: int = 15
    graph_search_top_k: int = 10
    rerank_top_k: int = 5

    # Temporal decay (half-life in days) by utility grade
    # Higher utility = longer retention, core facts never decay
    temporal_decay_high: int = 180    # 6 months for HIGH utility
    temporal_decay_medium: int = 60   # 2 months for MEDIUM
    temporal_decay_low: int = 14      # 2 weeks for LOW
    temporal_decay_core: int = 0      # No decay for core facts (0 = disabled)

    # Chunking
    similarity_threshold: float = 0.85
    min_chunk_sentences: int = 3
    max_chunk_tokens: int = 512

    # Voice I/O (v1.2.0+)
    # TTS (Text-to-Speech)
    tts_enabled: bool = False
    tts_voice: str = "af_heart"  # Kokoro voice: af_heart (A-grade), af_bella, af_nicole, etc.
    tts_speed: float = 1.0       # Speech speed multiplier (0.5-2.0)

    # STT (Speech-to-Text) - Not yet implemented
    stt_enabled: bool = False
    whisper_model: str = "medium"
    wake_word: str = "hey assistant"

    class Config:
        env_file = ".env"


settings = Settings()
