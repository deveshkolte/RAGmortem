"""Module execution entrypoint: python -m ragmortem."""

from __future__ import annotations

import sys
from ragmortem.cli import main

if __name__ == "__main__":
    sys.exit(main())
