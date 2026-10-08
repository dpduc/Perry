"""Where Perry keeps files outside the repo."""

import os
import sys
from pathlib import Path


def app_data_dir() -> Path:
    """Perry's folder in the user's app-data (%APPDATA%\\Perry on Windows)."""
    if override := os.getenv("PERRY_APP_DATA_DIR"):
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Perry"
    base = os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / "perry"
