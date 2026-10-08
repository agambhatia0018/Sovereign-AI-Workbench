"""Central settings. Values come from environment variables or a .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"

    # RAG settings
    data_dir: str = "data"          # uploads + vector DB live here (git-ignored)
    chunk_size: int = 800           # characters per chunk
    chunk_overlap: int = 150        # overlap so sentences are not cut off
    top_k: int = 4                  # how many chunks to give the LLM

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
