"""vocab <step> — command-line entry point."""
import argparse
import json


def cmd_setup(args):
    from . import parity
    print(json.dumps(parity.run(), indent=1))


def cmd_notes(args):
    from . import modelnotes
    out = modelnotes.run()
    print(json.dumps({m: out[m]["timing"] for m in ("gemma", "clap")}, indent=1))


def cmd_plan(args):
    from . import plan
    plan.run()
    print("reports/phase2_plan.json, data/selection/*.parquet")


def cmd_fetch(args):
    from . import fetch
    fetch.run(args.dataset)


def main():
    ap = argparse.ArgumentParser(prog="vocab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("setup", cmd_setup), ("notes", cmd_notes), ("plan", cmd_plan), ("fetch", cmd_fetch)]:
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name == "fetch":
            p.add_argument("dataset", choices=["esc50", "fsd50k", "cremad", "xenocanto", "fma", "musopen", "all"])
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
