"""
No secrets in tracked files — a scan of everything git tracks for API-key /
access-key patterns. (.env is gitignored and not scanned.) Generic to any
repo, not specific to the generator itself.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = [
    re.compile(r"sk-" + r"[A-Za-z0-9_\-]{20,}"),        # OpenAI-style
    re.compile(r"AKIA" + r"[0-9A-Z]{16}"),               # AWS access key id
    re.compile(r"(?i)(secret|token|password|api[_-]?key)\s*[:=]\s*"
               r"['\"][A-Za-z0-9/+_\-]{24,}['\"]"),
]
SKIP_EXT = {".parquet", ".pdf", ".png", ".ico", ".jpg", ".jpeg", ".gz", ".pyc", ".so"}


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [l for l in out.stdout.splitlines() if l.strip()]


def test_no_secret_patterns_in_tracked_files():
    offenders = []
    for rel in _tracked_files():
        if Path(rel).suffix.lower() in SKIP_EXT:
            continue
        fp = REPO_ROOT / rel
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError):
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                offenders.append((rel, pat.pattern))
    assert not offenders, f"possible secrets in tracked files: {offenders}"
