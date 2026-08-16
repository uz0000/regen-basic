"""
The CLI's exit codes are an interface, so they are tested like one.

A failed check is not a crash — the tool ran fine and is reporting that the
result is bad. That is the whole point of having checks, and a pipeline
watching only for crashes would sail straight past it. So "checks failed"
and "could not run" get different codes, and both differ from success.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
EXIT_OK, EXIT_CHECKS_FAILED, EXIT_CANNOT_RUN = 0, 1, 2


def _run(*args):
    """Invoke the CLI the way a script would, not by importing it."""
    return subprocess.run(
        [sys.executable, "-m", "cli.main", *args],
        cwd=REPO, capture_output=True, text=True,
    )


@pytest.fixture
def passing_table(tmp_path):
    """Correlated numerics only — the simulator reproduces this faithfully."""
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(50, 10, n)
    df = pd.DataFrame({"x": x, "y": x * 0.8 + rng.normal(0, 3, n),
                       "flag": (x > 50).astype(int)})
    p = tmp_path / "passing.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def failing_table(tmp_path):
    """A category whose effect on a measurement does not run in alphabetical
    order — a relationship the simulator cannot represent and the checks are
    expected to catch."""
    rng = np.random.default_rng(0)
    n = 1500
    cat = rng.choice(["alpha", "beta", "gamma"], n)
    base = np.select([cat == "alpha", cat == "beta"], [30.0, 10.0], default=20.0)
    df = pd.DataFrame({"cat": cat, "val": base + rng.normal(0, 3, n)})
    p = tmp_path / "failing.csv"
    df.to_csv(p, index=False)
    return p


def test_success_exits_zero(passing_table, tmp_path):
    r = _run("generate", str(passing_table), "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_OK, r.stderr


def test_failed_checks_exit_one(failing_table, tmp_path):
    r = _run("generate", str(failing_table), "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_CHECKS_FAILED, r.stdout + r.stderr


def test_failed_checks_still_write_the_data(failing_table, tmp_path):
    """A failed check is a verdict, not an abort — you need the output to see
    what went wrong."""
    out = tmp_path / "o"
    _run("generate", str(failing_table), "--out", str(out))
    written = out / "failing_synthetic.csv"
    assert written.exists()
    assert len(pd.read_csv(written)) > 0


def test_allow_fail_exits_zero_on_failed_checks(failing_table, tmp_path):
    r = _run("generate", str(failing_table), "--out", str(tmp_path / "o"), "--allow-fail")
    assert r.returncode == EXIT_OK, r.stdout + r.stderr


def test_missing_file_exits_two(tmp_path):
    r = _run("generate", str(tmp_path / "does_not_exist.csv"), "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_CANNOT_RUN


def test_unusable_input_exits_two(tmp_path):
    """Input the simulator refuses (here: every column an identifier) is a
    cannot-run, distinct from a check that ran and failed."""
    p = tmp_path / "ids.csv"
    pd.DataFrame({"id": np.arange(1, 201),
                  "uuid": [f"u-{i}" for i in range(200)]}).to_csv(p, index=False)
    r = _run("generate", str(p), "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_CANNOT_RUN


def test_failure_is_explained_on_stderr(failing_table, tmp_path):
    """Whoever reads the log should learn which table failed and that the
    data still exists."""
    r = _run("generate", str(failing_table), "--out", str(tmp_path / "o"))
    assert "failing" in r.stderr
    assert "--allow-fail" in r.stderr


def test_one_failure_among_several_fails_the_run(passing_table, failing_table, tmp_path):
    """A mixed batch must not be reported as fine because most of it was."""
    r = _run("generate", str(passing_table), str(failing_table),
             "--out", str(tmp_path / "o"))
    assert r.returncode == EXIT_CHECKS_FAILED
