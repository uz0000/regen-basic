"""Add repo root to sys.path so engine/, contracts/, simulate/ are importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
