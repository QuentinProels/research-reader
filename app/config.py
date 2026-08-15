from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8090
    data_dir: Path = Path("./data")

    app_password_hash: str = ""
    session_secret: str = "change-me"

    llm_base_url: str = "http://localhost:8084/v1"
    llm_api_key: str = ""
    llm_model: str = "Qwen-Coder"

    kokoro_model_path: Path = Path("./models/kokoro-v1.0.onnx")
    kokoro_voices_path: Path = Path("./models/voices-v1.0.bin")
    kokoro_voice: str = "af_heart"

    max_upload_bytes: int = 100 * 1024 * 1024
    max_pages: int = 500
    parse_timeout_seconds: int = 120

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"


settings = Settings()
settings.papers_dir.mkdir(parents=True, exist_ok=True)
