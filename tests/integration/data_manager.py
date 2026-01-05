# Minimal DataManager stub for integration tests to satisfy conftest imports.
from pathlib import Path

class DataManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, content: bytes) -> None:
        p = self.data_dir / name
        p.write_bytes(content)

    def load(self, name: str) -> bytes:
        p = self.data_dir / name
        return p.read_bytes() if p.exists() else b""
