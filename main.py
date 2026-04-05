"""Application entry point."""

from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from this project folder (not the shell's current directory).
load_dotenv(Path(__file__).resolve().parent / ".env")

# Re-export ASGI app for `uvicorn main:app` (Codespaces / CLI).
from src.api.app import app, run


def main() -> None:
    """Run the application."""
    run()


if __name__ == "__main__":
    main()
