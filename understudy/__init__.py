"""Understudy — a negotiation agent measured against fitted seller personas."""

import os
import re
import warnings
from pathlib import Path

from dotenv import load_dotenv

_ENV = Path(__file__).resolve().parent.parent / ".env"
# Names long enough to look like a pasted secret rather than a typo'd word.
_SUSPECT = re.compile(r"^([A-Z][A-Z0-9_]{4,})(?!=)([^\s=]{12,})$")


def _warn_on_malformed(path: Path) -> list[str]:
    """Catch `NAME<value>` lines, where the '=' was lost in a paste.

    dotenv treats the whole line as a variable *name* with an empty value, so
    the key silently does not exist and the feature it gates looks broken —
    while the secret sits in an environment variable named after itself.
    """
    if not path.exists():
        return []
    bad = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if line.startswith("#") or "=" in line or not line:
            continue
        if _SUSPECT.match(line):
            bad.append(f"{path.name} line {n}: missing '=' after the variable name")
    for message in bad:
        warnings.warn(f"malformed env file — {message}", RuntimeWarning, stacklevel=2)
    return bad


_warn_on_malformed(_ENV)

def _drop_blanks(path: Path) -> list[str]:
    """A blank line in a template means "unset", not "empty string".

    `.env.example` ships every variable blank, so copying it leaves entries
    like `OPENAI_BASE_URL=` behind. Third-party SDKs read those variables
    directly and an empty string is not falsy to them: the OpenAI client
    dutifully used "" as its base URL and failed with a connection error that
    named neither the variable nor the file. Same shape as a blank model id.
    """
    if not path.exists():
        return []
    cleared = []
    for line in path.read_text().splitlines():
        name, sep, _ = line.strip().partition("=")
        if sep and name and os.environ.get(name) == "":
            del os.environ[name]
            cleared.append(name)
    return cleared


# Loaded once, on first import of anything in the package, so the CLI, the API
# and a bare REPL all see the same configuration. override=False means a real
# environment variable always beats the file.
load_dotenv(override=False)
_drop_blanks(_ENV)
