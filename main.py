"""Application entry point."""

from pathlib import Path

from dotenv import load_dotenv

# Load `.env` from this project folder (not the shell's current directory).
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.api.app import run


def main() -> None:
    """Run the application."""
    run()


if __name__ == "__main__":
    main()


