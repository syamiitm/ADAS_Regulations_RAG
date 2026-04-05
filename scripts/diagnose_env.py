"""Print which variable names exist in .env (no values). Run from project root:

    python scripts/diagnose_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import dotenv_values

root = Path(__file__).resolve().parent.parent
env_path = root / ".env"
print(f"path: {env_path}")
print(f"exists: {env_path.is_file()}")
if not env_path.is_file():
    sys.exit(1)

for enc in ("utf-8-sig", "utf-8", "utf-16-le", "latin-1"):
    try:
        vals = dotenv_values(str(env_path), encoding=enc)
    except (UnicodeError, UnicodeDecodeError):
        continue
    keys = [k for k, v in vals.items() if k and v and str(v).strip()]
    print(f"encoding {enc!r}: {len(keys)} non-empty keys -> {sorted(keys)}")
    if keys:
        break
else:
    print("Could not parse .env with common encodings.")
