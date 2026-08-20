"""`.env.example` must document every variable the code actually reads.

A key that exists only in someone's shell is a configuration bug waiting to
happen, and an undocumented one is worse: the feature it gates looks broken.
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / ".env.example"

# Read by the anthropic SDK itself rather than by our code, so it never appears
# in a grep of understudy/.
SDK_READ = {"ANTHROPIC_BASE_URL"}


def env_vars_used_in_code() -> set[str]:
    pattern = re.compile(r'environ(?:\.get)?\(\s*"([A-Z_]+)"')
    found: set[str] = set()
    for path in (ROOT / "understudy").rglob("*.py"):
        found |= set(pattern.findall(path.read_text()))
    return found


def documented_in_example() -> set[str]:
    return set(re.findall(r"^([A-Z_]+)=", EXAMPLE.read_text(), re.M))


def test_example_file_exists():
    assert EXAMPLE.exists()


def test_every_variable_the_code_reads_is_documented():
    missing = env_vars_used_in_code() - documented_in_example()
    assert not missing, f"undocumented in .env.example: {sorted(missing)}"


def test_no_stale_variables_documented():
    stale = documented_in_example() - env_vars_used_in_code() - SDK_READ
    assert not stale, f"documented but never read: {sorted(stale)}"


def test_example_holds_no_real_secrets():
    """The template is committed; it must never carry a filled-in key."""
    for line in EXAMPLE.read_text().splitlines():
        if re.match(r"^[A-Z_]+=", line):
            key, _, value = line.partition("=")
            assert value in ("", "false"), f"{key} has a value in the committed template"


def test_dotenv_is_actually_loaded():
    import understudy
    assert "load_dotenv" in Path(understudy.__file__).read_text()


def test_malformed_env_lines_are_reported(tmp_path):
    """`OPENAI_API_KEY sk-...` with no '=' must not fail silently.

    dotenv parses such a line as a variable *name*, so the key appears unset
    and the feature it gates looks broken for reasons nothing explains.
    """
    from understudy import _warn_on_malformed

    env = tmp_path / ".env"
    env.write_text(
        "GOOD_KEY=value\n"
        "# a comment\n"
        "\n"
        "OPENAI_API_KEYEXAMPLE-NOT-A-REAL-KEY-000000\n"
    )
    problems = _warn_on_malformed(env)
    assert len(problems) == 1
    assert "line 4" in problems[0]


def test_well_formed_env_files_report_nothing(tmp_path):
    from understudy import _warn_on_malformed

    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=EXAMPLE-NOT-A-REAL-KEY-000000\nEMPTY=\n# note\n")
    assert _warn_on_malformed(env) == []


def test_blank_model_var_falls_back_to_a_usable_default(monkeypatch):
    """A template ships `OPENAI_MODEL=`, which is "" — not a usable model id."""
    from understudy.llm import FALLBACK_OPENAI_MODEL, default_openai_model

    monkeypatch.setenv("OPENAI_MODEL", "")
    assert default_openai_model() == FALLBACK_OPENAI_MODEL

    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert default_openai_model() == FALLBACK_OPENAI_MODEL

    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")
    assert default_openai_model() == "gpt-5.4"


def test_blank_entries_are_unset_not_empty_strings(tmp_path, monkeypatch):
    """`OPENAI_BASE_URL=` must leave the variable absent.

    SDKs read these variables themselves and treat "" as a real value: the
    OpenAI client used "" as its base URL and failed with a bare connection
    error naming neither the variable nor the file.
    """
    from understudy import _drop_blanks

    env = tmp_path / ".env"
    env.write_text("OPENAI_BASE_URL=\nOPENAI_MODEL=gpt-5.4\n# comment\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4")

    cleared = _drop_blanks(env)
    assert cleared == ["OPENAI_BASE_URL"]
    assert "OPENAI_BASE_URL" not in os.environ
    assert os.environ["OPENAI_MODEL"] == "gpt-5.4"


def test_a_real_value_is_never_dropped(tmp_path, monkeypatch):
    from understudy import _drop_blanks

    env = tmp_path / ".env"
    env.write_text("OPENAI_BASE_URL=https://example.test/v1\n")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    assert _drop_blanks(env) == []
    assert os.environ["OPENAI_BASE_URL"] == "https://example.test/v1"
