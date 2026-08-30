"""``python -m itsbob`` — the same entry point as the ``itsbob`` command.

Useful when the console script is not on PATH, which is the normal state
inside a virtualenv that has not been activated.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
