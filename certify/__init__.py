"""
certify — check whether a conclusion survives the round trip.

Fits a declared analysis on both the real and the synthetic table and reports,
per coefficient, whether they agree. Independent of whatever produced the
synthetic data: it never asks where the rows came from, so it applies equally
to simulate/'s output and to any other tool's.
"""

from certify.certifier import certify_dataset, certify_many

__all__ = ["certify_dataset", "certify_many"]
