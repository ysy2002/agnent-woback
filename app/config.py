from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    app_env: str = "development"
    log_level: str = "INFO"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # Tickets with confidence >= this threshold are auto-replied
    auto_reply_threshold: float = 0.80


settings = Settings()
