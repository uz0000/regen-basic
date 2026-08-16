"""
simulate — build a synthetic stand-in for a real table.

Models the table's joint distribution (what each column looks like *and* how
the columns move together) and draws fresh rows from it. See certify/ for
the independent check of whether a conclusion drawn from those rows matches
the one the real data would have given.
"""

from simulate.generate import SimulationResult, generate_table, generate_tables

__all__ = ["SimulationResult", "generate_table", "generate_tables"]
