"""
Engine — the pieces basic/generate.py composes into a table generator.

Pure Python. No LLM client, no agent framework, no network library.

  engine.ingest    — column-type/identifier inference (schema, no label needed)
  engine.prior     — the mixed-data Gaussian copula sampler
  engine.auditor   — fidelity checks (per-column + cross-column correlation)
  engine.privacy   — the verbatim-duplicate guard
"""
