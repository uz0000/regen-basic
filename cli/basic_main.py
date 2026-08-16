"""
synth — CLI for the basic generic table generator (basic/generate.py).

Separate from `regen` (cli/main.py) on purpose: that CLI's flags are all
rare-event/estimand concepts (--label, --rare-mode, --scenario). This one
takes any number of tables and generates a synthetic version of each, no
label column or declared analysis required.

Usage:
    synth generate table1.csv table2.csv ... --n-rows 500 --out synth-output/
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    if args.command == "generate":
        _cmd_generate(args)


def _build_parser():
    p = argparse.ArgumentParser(prog="synth", description="Generic synthetic table generator")
    p.add_argument("--version", "-V", action="version", version=f"synth {__version__}")
    sub = p.add_subparsers(dest="command")

    gen_p = sub.add_parser("generate", help="Generate a synthetic version of one or more tables")
    gen_p.add_argument("tables", nargs="+", help="Path(s) to input table(s) — CSV")
    gen_p.add_argument("--n-rows", type=int, default=None,
                       help="Rows per synthetic table (default: same as the real table)")
    gen_p.add_argument("--out", type=str, default="synth-output",
                       help="Output directory (default: ./synth-output)")
    gen_p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    gen_p.add_argument("--json", action="store_true", help="Print machine-readable summary")
    gen_p.set_defaults(command="generate")

    return p


def _cmd_generate(args):
    from basic.generate import generate_table

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for path_str in args.tables:
        path = Path(path_str)
        if not path.exists():
            print(f"[synth] ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        real_df = pd.read_csv(path)
        n_rows = args.n_rows if args.n_rows is not None else len(real_df)
        table_name = path.stem
        try:
            result = generate_table(
                real_df, n_rows, seed=args.seed, table_name=table_name,
            )
        except ValueError as e:
            print(f"[synth] ERROR generating {table_name}: {e}", file=sys.stderr)
            sys.exit(1)
        out_path = out_dir / f"{table_name}_synthetic.csv"
        result.synthetic_df.to_csv(out_path, index=False)
        results[table_name] = result
        if not args.json:
            print(result.summary())
            print(f"  -> {out_path}\n")

    if args.json:
        payload = {
            name: {
                "n_real_rows": r.n_real_rows,
                "n_synthetic_rows": r.n_synthetic_rows,
                "fidelity_passed": r.fidelity_passed,
                "correlation_delta": r.correlation_delta,
                "columns_failed": [c.col for c in r.column_reports if not c.passed],
                "n_duplicates_guarded": r.n_duplicates_guarded,
                "identifier_cols": r.identifier_cols,
                "out_path": str(out_dir / f"{name}_synthetic.csv"),
            }
            for name, r in results.items()
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
