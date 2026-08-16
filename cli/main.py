"""
synth — command-line interface to the table simulator (simulate/generate.py).

Takes any number of real tables and writes a synthetic stand-in for each,
reporting what was checked about every one.

Usage:
    synth generate table1.csv table2.csv ... --n-rows 500 --out synth-output/

Exit codes, so a script can tell the two kinds of bad news apart:

    0   every table simulated and passed every check
    1   at least one table FAILED a check — the data was still written, so
        you can look at what went wrong, but nothing downstream should treat
        it as a faithful stand-in without deciding to
    2   could not run: a missing file, or input the simulator refuses
        (no output written)

The distinction matters because a failed check is not a crash. The tool did
its job and is telling you the answer is bad, which is the entire point of
having the checks — and a pipeline that only looked for a crash would sail
straight past it. Pass --allow-fail to treat a failed check as exit 0 when
you genuinely want the data anyway.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

__version__ = "0.1.0"

EXIT_OK = 0
EXIT_CHECKS_FAILED = 1
EXIT_CANNOT_RUN = 2


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(EXIT_CANNOT_RUN)
    if args.command == "generate":
        sys.exit(_cmd_generate(args))


def _build_parser():
    p = argparse.ArgumentParser(
        prog="synth",
        description="Simulate a table: build a synthetic stand-in for real data "
                    "that keeps each column's distribution and how the columns "
                    "move together.")
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
    gen_p.add_argument("--allow-fail", action="store_true",
                       help="Exit 0 even when a table fails its checks "
                            "(default: exit 1 so a script notices)")
    gen_p.set_defaults(command="generate")

    return p


def _cmd_generate(args):
    from simulate.generate import generate_table

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for path_str in args.tables:
        path = Path(path_str)
        if not path.exists():
            print(f"[synth] ERROR: file not found: {path}", file=sys.stderr)
            return EXIT_CANNOT_RUN
        real_df = pd.read_csv(path)
        n_rows = args.n_rows if args.n_rows is not None else len(real_df)
        table_name = path.stem
        try:
            result = generate_table(
                real_df, n_rows, seed=args.seed, table_name=table_name,
            )
        except ValueError as e:
            print(f"[synth] ERROR generating {table_name}: {e}", file=sys.stderr)
            return EXIT_CANNOT_RUN
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
                "categorical_association_delta": r.categorical_association_delta,
                "categorical_worst_pair": r.categorical_worst_pair,
                "columns_failed": [c.col for c in r.column_reports if not c.passed],
                "n_duplicates_guarded": r.n_duplicates_guarded,
                "identifier_cols": r.identifier_cols,
                "out_path": str(out_dir / f"{name}_synthetic.csv"),
            }
            for name, r in results.items()
        }
        print(json.dumps(payload, indent=2))

    failed = [name for name, r in results.items() if not r.fidelity_passed]
    if failed and not args.allow_fail:
        if not args.json:
            print(f"[synth] {len(failed)} of {len(results)} table(s) failed their "
                  f"checks: {', '.join(failed)}", file=sys.stderr)
            print("[synth] the data was still written — inspect it, or pass "
                  "--allow-fail to exit 0 anyway", file=sys.stderr)
        return EXIT_CHECKS_FAILED
    return EXIT_OK


if __name__ == "__main__":
    main()
