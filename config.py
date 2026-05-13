from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    repo_path: str = "./sample_repo"
    db_path: str = "./index.db"
    vector_dir: str = "./vectors"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 50
    top_k: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
