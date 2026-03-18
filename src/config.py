# src/config.py
# Centralized configuration using Pydantic Settings.
# Reads values from environment variables or a .env file.
# All other modules import from here — never from os.environ directly.

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic validates each field's type automatically.
    If a required variable is missing, the app fails immediately
    with a clear error — not silently with a wrong value.
    """

    # --- LLM settings ---
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL where Ollama is running"
    )
    ollama_model: str = Field(
        default="mistral",
        description="Name of the Ollama model to use"
    )

    # --- Embedding settings ---
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model name for generating embeddings"
    )

    # --- Vector database settings ---
    chroma_persist_directory: str = Field(
        default="./vectorstore",
        description="Filesystem path where ChromaDB stores its data"
    )
    chroma_collection_name: str = Field(
        default="docs",
        description="ChromaDB collection to store document embeddings"
    )

    # --- Document ingestion settings ---
    data_directory: str = Field(
        default="./data/raw",
        description="Directory containing source documentation files"
    )
    chunk_size: int = Field(
        default=512,
        description="Number of characters per text chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Character overlap between consecutive chunks"
    )

    # --- API settings ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_debug: bool = Field(default=False)

    # Pydantic reads .env file automatically
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,      # OLLAMA_MODEL and ollama_model both work
    )


# Module-level singleton — import this everywhere
# This is the standard Python pattern for shared config
settings = Settings()