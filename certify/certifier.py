"""
Generator-agnostic estimand certification.

Given a real reference dataset, *any* synthetic dataset (however it was
produced), and a declared analysis (``EstimandSpec``), decide whether the
synthetic data preserves the estimate: fit the analysis on both, compare
each coefficient, emit a portable, recomputable certificate. The synthetic
data's provenance is irrelevant — only whether the conclusion survives.

The certificate carries theta_real +/- SE (a disclosed aggregate), so a
third party can recompute theta_synth from the synthetic data alone and
re-check the verdict without the real rows.

Deterministic, no LLM, numpy + scipy only (via certify.estimand).
"""

from __future__ import annotations

from typing import Any, Dict

from contracts.types import EstimandSpec
from certify.estimand import evaluate, reference_aggregate

METRIC_VERSION = 1  # bump if the certification rule/statistic changes


def certify_dataset(real_df, synthetic_df, estimand: EstimandSpec,
                    source: str = "") -> Dict[str, Any]:
    """Certify whether ``synthetic_df`` preserves ``estimand`` measured on ``real_df``.

    Returns a portable certificate: the verdict (``certified`` / ``status``),
    per-coefficient theta_real vs theta_synth with the consistency test, the
    rule + CI level, the metric version, the source label, and the disclosed
    theta_real +/- SE (so the certificate is re-checkable against the
    synthetic data alone). Never raises — an unfittable spec becomes an
    honest ``uncertifiable`` status.
    """
    assessment, real_fit = evaluate(real_df, synthetic_df, estimand)
    cert = dict(assessment)
    cert["source"] = source
    cert["metric"] = "estimand_delta"
    cert["metric_version"] = METRIC_VERSION
    if real_fit is not None:
        cert["theta_real_disclosed"] = reference_aggregate(real_fit, estimand)
    return cert


def certify_many(real_df, synthetics: Dict[str, Any],
                 estimand: EstimandSpec) -> Dict[str, Dict[str, Any]]:
    """Certify the *same* estimand across many synthetic sources -> {name: certificate}.

    theta_real is identical across all of them (it comes from real_df); what
    varies is theta_synth per source. This is the generator-agnostic
    comparison: it is not about who made the data, only whether each
    preserves the conclusion.
    """
    return {name: certify_dataset(real_df, df, estimand, source=name)
            for name, df in synthetics.items()}
