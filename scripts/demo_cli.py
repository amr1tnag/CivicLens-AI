#!/usr/bin/env python3
"""Command-line demo: analyze a complaint without running the API/DB.

Usage:
    python scripts/demo_cli.py "There is a huge pothole on MG Road, a bike
    already crashed into it, please fix immediately!"

    python scripts/demo_cli.py --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from civiclens.config import CATEGORY_MODEL_PATH, URGENCY_MODEL_PATH  # noqa: E402
from civiclens.pipeline import ComplaintAnalyzer  # noqa: E402

SAMPLE_COMPLAINTS = [
    "There is a massive pothole on MG Road and a two-wheeler already crashed into it. Someone could get seriously injured, please fix immediately!",
    "The park in Ward 6 - Greenfield needs its grass cut, it has been overgrown for a while. No rush, just flagging it.",
    "No water supply at all in Ward 2 - Old Town for the past two days, elderly residents are struggling. Please treat this as an emergency.",
    "One streetlight near Elm Street has been out for a few days, the rest of the street is fine.",
]


def print_ticket(text: str, analyzer: ComplaintAnalyzer) -> None:
    ticket = analyzer.analyze(text)
    print("-" * 70)
    print(f"Complaint: {text}")
    print()
    print(json.dumps(asdict(ticket), indent=2, default=str))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="CivicLens AI CLI demo")
    parser.add_argument("text", nargs="?", help="Complaint text to analyze")
    parser.add_argument("--interactive", action="store_true", help="Prompt for complaints in a loop")
    parser.add_argument("--samples", action="store_true", help="Run a handful of built-in sample complaints")
    args = parser.parse_args()

    if not CATEGORY_MODEL_PATH.exists() or not URGENCY_MODEL_PATH.exists():
        print("Models not found. Train them first:")
        print("  python -m civiclens.data.generate_dataset")
        print("  python -m civiclens.train")
        sys.exit(1)

    analyzer = ComplaintAnalyzer.load()

    if args.samples:
        for text in SAMPLE_COMPLAINTS:
            print_ticket(text, analyzer)
        return

    if args.interactive:
        print("CivicLens AI - interactive demo. Type a complaint and press enter (Ctrl+D to quit).")
        while True:
            try:
                text = input("\n> ").strip()
            except EOFError:
                break
            if text:
                print_ticket(text, analyzer)
        return

    if args.text:
        print_ticket(args.text, analyzer)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
