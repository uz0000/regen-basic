"""
Tests for basic/generate.py — the generic (no label, no rare-event split)
table generator. Exercises continuous, categorical, binary, and identifier
columns together, since the ordering bug between copula-encoding and the
fidelity/privacy checks only shows up when a categorical column is present.
"""

import numpy as np
import pandas as pd
import pytest

from basic.generate import BasicResult, generate_table, generate_tables


def _mixed_real_df(n=600, seed=0):
    """A table with continuous (correlated), categorical, binary, and an
    identifier column — one of every kind basic/generate.py has to handle."""
    rng = np.random.default_rng(seed)
    income = rng.normal(60000, 15000, n).clip(15000, None)
    # debt correlated with income so the correlation-preservation check is meaningful
    debt = income * 0.3 + rng.normal(0, 2000, n)
    region = rng.choice(["north", "south", "east", "west"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    approved = (income > 55000).astype(int)  # binary, correlated with income
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "income": income,
        "debt": debt,
        "region": region,
        "approved": approved,
    })


def test_generates_requested_row_count():
    real = _mixed_real_df()
    result = generate_table(real, n_rows=300, seed=1)
    assert len(result.synthetic_df) == 300
    assert result.n_real_rows == len(real)


def test_columns_and_dtypes_preserved():
    real = _mixed_real_df()
    result = generate_table(real, n_rows=200, seed=1)
    synth = result.synthetic_df
    assert list(synth.columns) == list(real.columns)
    # categorical values must be real categories, not leftover numeric codes
    assert set(synth["region"].unique()) <= set(real["region"].unique())
    # binary column only takes the two observed values
    assert set(synth["approved"].unique()) <= set(real["approved"].unique())
    # continuous stays numeric and in range
    assert synth["income"].between(real["income"].min(), real["income"].max()).all()


def test_identifier_column_reminted_not_duplicated():
    real = _mixed_real_df()
    result = generate_table(real, n_rows=200, seed=1)
    synth = result.synthetic_df
    assert result.identifier_cols == ["customer_id"]
    # fresh, unique, and past the real max — never collides with a real id
    assert synth["customer_id"].is_unique
    assert synth["customer_id"].min() > real["customer_id"].max()


def test_correlation_structure_is_preserved_not_just_marginals():
    real = _mixed_real_df(n=2000)
    result = generate_table(real, n_rows=2000, seed=1)
    real_corr = real[["income", "debt"]].corr().iloc[0, 1]
    synth_corr = result.synthetic_df[["income", "debt"]].corr().iloc[0, 1]
    # income/debt are constructed to correlate ~0.9; an independent-per-column
    # sampler would produce ~0 here — this is the check that would have caught
    # the certifier-demo failure mode (marginals right, joint structure wrong).
    assert real_corr > 0.7
    assert synth_corr > 0.7


def test_fidelity_report_present_and_reasonable():
    real = _mixed_real_df(n=1000)
    result = generate_table(real, n_rows=1000, seed=1)
    assert isinstance(result.fidelity_passed, bool)
    assert len(result.column_reports) == 4  # all cols except the identifier
    assert result.correlation_delta is not None


def test_duplicate_guard_count_is_reported():
    real = _mixed_real_df(n=500)
    result = generate_table(real, n_rows=500, seed=1)
    assert isinstance(result.n_duplicates_guarded, int)
    assert result.n_duplicates_guarded >= 0


def test_no_synthetic_row_is_a_verbatim_real_row():
    real = _mixed_real_df(n=300)
    result = generate_table(real, n_rows=300, seed=1)
    synth = result.synthetic_df.drop(columns=["customer_id"])
    real_no_id = real.drop(columns=["customer_id"])
    merged = synth.merge(real_no_id, how="inner")
    assert len(merged) == 0, "a synthetic row exactly reproduced a real row's attributes"


def test_deterministic_same_seed_same_output():
    real = _mixed_real_df()
    a = generate_table(real, n_rows=150, seed=7)
    b = generate_table(real, n_rows=150, seed=7)
    pd.testing.assert_frame_equal(a.synthetic_df, b.synthetic_df)


def test_different_seed_different_output():
    real = _mixed_real_df()
    a = generate_table(real, n_rows=150, seed=7)
    b = generate_table(real, n_rows=150, seed=8)
    assert not a.synthetic_df.equals(b.synthetic_df)


def test_generate_tables_handles_multiple_tables_independently():
    tables = {"customers": _mixed_real_df(seed=1), "customers_v2": _mixed_real_df(seed=2)}
    results = generate_tables(tables, n_rows=100, seed=42)
    assert set(results.keys()) == {"customers", "customers_v2"}
    for name, r in results.items():
        assert isinstance(r, BasicResult)
        assert len(r.synthetic_df) == 100
        assert r.table_name == name


def test_generate_tables_per_table_row_counts():
    tables = {"a": _mixed_real_df(seed=1), "b": _mixed_real_df(seed=2)}
    results = generate_tables(tables, n_rows={"a": 50, "b": 75}, seed=42)
    assert len(results["a"].synthetic_df) == 50
    assert len(results["b"].synthetic_df) == 75


def test_all_categorical_table_does_not_crash():
    real = pd.DataFrame({
        "color": np.random.default_rng(0).choice(["red", "green", "blue"], size=200),
        "size": np.random.default_rng(1).choice(["S", "M", "L"], size=200),
    })
    result = generate_table(real, n_rows=100, seed=1)
    assert len(result.synthetic_df) == 100
    assert set(result.synthetic_df["color"].unique()) <= {"red", "green", "blue"}


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        generate_table(pd.DataFrame({"a": []}), n_rows=10)


def test_zero_rows_requested_raises():
    real = _mixed_real_df()
    with pytest.raises(ValueError):
        generate_table(real, n_rows=0)


def test_all_identifier_columns_raises():
    real = pd.DataFrame({
        "id": np.arange(1, 101),
        "uuid": [f"u-{i}" for i in range(100)],
    })
    with pytest.raises(ValueError, match="identifier"):
        generate_table(real, n_rows=10)


def test_missing_values_raise_loudly_not_silently():
    real = _mixed_real_df(n=50)
    real.loc[0, "income"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        generate_table(real, n_rows=10)
