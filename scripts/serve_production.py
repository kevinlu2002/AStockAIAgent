from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from waitress import serve

from ashare_ai_agent.webapp import create_app


def main() -> None:
    host = os.environ.get("ASHARE_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("ASHARE_PORT", "7860")))
    threads = int(os.environ.get("ASHARE_THREADS", "8"))
    serve(create_app(), host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
