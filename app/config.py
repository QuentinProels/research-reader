from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_host: str = "127.0.0.1"
    app_port: int = 8090
    data_dir: Path = Path("./data")

    # A bcrypt hash contains '$', and `docker compose` interpolates '$VAR' inside every
    # value of the .env it auto-loads -- whether or not the compose file uses it. Keeping
    # the hash in a file sidesteps that entirely; .env holds the path, not the hash.
    app_password_hash_file: Path = Path("./secrets/app_password.hash")
    session_secret: str = "change-me"

    database_url: str = "postgresql://reader:reader@127.0.0.1:5433/reader"

    llm_base_url: str = "http://localhost:8084/v1"
    llm_api_key: str = ""
    llm_model: str = "Qwen-Coder"

    kokoro_model_path: Path = Path("./models/kokoro-v1.0.onnx")
    kokoro_voices_path: Path = Path("./models/voices-v1.0.bin")
    kokoro_voice: str = "af_heart"

    # faster-whisper on CPU: CTranslate2 has no ROCm backend, and both GPUs hold the 35B.
    stt_model: str = "tiny.en"  # measured 1.8s vs 5.4s for small.en; errors are formatting, not meaning

    max_upload_bytes: int = 100 * 1024 * 1024
    max_pages: int = 500
    parse_timeout_seconds: int = 120

    @property
    def models_dir(self) -> Path:
        return self.kokoro_model_path.parent

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def password_hash(self) -> str:
        """Empty means the password wall is off -- Cloudflare Access is then the only lock."""
        if self.app_password_hash_file.exists():
            return self.app_password_hash_file.read_text().strip()
        return ""


settings = Settings()
settings.papers_dir.mkdir(parents=True, exist_ok=True)
