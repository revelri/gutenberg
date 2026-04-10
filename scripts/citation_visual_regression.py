#!/usr/bin/env python3
"""Citation style visual regression test.

Generates formatted citations for all 6 styles and compares against golden
reference outputs. First run with --init to bootstrap the golden file.
Subsequent runs diff against it; exit 1 on regression.

Usage:
    python scripts/citation_visual_regression.py --init      # create golden
    python scripts/citation_visual_regression.py --update    # regenerate golden
    python scripts/citation_visual_regression.py [--verbose] # regression check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "api"))

from core.citation_formatter import Citation, Style, format_bibliography, format_inline

GOLDEN_PATH = Path("data/eval/citation_golden.json")

STYLES = list(Style)

TEST_CITATIONS: dict[str, Citation] = {
    "single_author": Citation(
        quote="Every organism is a deterritorialization of the old.",
        author="Deleuze, Gilles",
        title="Difference and Repetition",
        year=1968,
        page=1,
        publisher="Columbia University Press",
        city="New York",
    ),
    "multi_author": Citation(
        quote="A rhizome has no beginning or end.",
        author="Deleuze, Gilles and Felix Guattari",
        title="A Thousand Plateaus",
        year=1980,
        page=24,
        publisher="University of Minnesota Press",
        city="Minneapolis",
    ),
    "three_plus_authors": Citation(
        quote="Collaborative thought is a multiplicity.",
        author="Deleuze, Gilles and Felix Guattari and Claire Parnet",
        title="Dialogues",
        year=1987,
        page=15,
        publisher="Columbia University Press",
        city="New York",
    ),
    "empty_author": Citation(
        quote="Unknown source text.",
        author="",
        title="Anonymous Work",
        year=2000,
        page=5,
        publisher="Unknown Press",
    ),
    "year_zero": Citation(
        quote="Undated passage.",
        author="Spinoza, Baruch",
        title="Ethics",
        year=0,
        page=42,
        publisher="Penguin Classics",
        city="London",
    ),
    "page_string": Citation(
        quote="EPUB chapter reference.",
        author="Deleuze, Gilles",
        title="The Logic of Sense",
        year=1969,
        page="ch. 3",
        publisher="Continuum",
        city="London",
    ),
}


def generate_output() -> dict:
    result: dict = {}
    for name, citation in TEST_CITATIONS.items():
        result[name] = {}
        for style in STYLES:
            key = style.value
            result[name][key] = {
                "inline": format_inline(citation, style),
                "bibliography": format_bibliography(citation, style),
            }
    return result


def print_table(output: dict) -> None:
    col_w = 18
    for test_name, styles in output.items():
        print(f"\n{'─' * 80}")
        print(f"  {test_name}")
        print(f"{'─' * 80}")
        header = f"{'Style':<12}{'Inline':<{col_w}}{'Bibliography'}"
        print(header)
        print(f"{'─' * 80}")
        for style_val, data in styles.items():
            bib = data["bibliography"]
            if len(bib) > 58:
                bib = bib[:55] + "..."
            print(f"{style_val:<12}{data['inline']:<{col_w}}{bib}")


def diff_outputs(current: dict, golden: dict) -> list[str]:
    diffs: list[str] = []
    for test_name in current:
        for style_val in current[test_name]:
            for fmt_type in ("inline", "bibliography"):
                cur = current[test_name][style_val][fmt_type]
                gld = golden.get(test_name, {}).get(style_val, {}).get(fmt_type, "")
                if cur != gld:
                    diffs.append(
                        f"  [{test_name}][{style_val}].{fmt_type}\n"
                        f"    expected: {gld}\n"
                        f"    got:      {cur}"
                    )
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="Citation visual regression test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--init", action="store_true", help="Bootstrap golden file")
    group.add_argument("--update", action="store_true", help="Regenerate golden file")
    parser.add_argument(
        "--verbose", action="store_true", help="Show full diff on regression"
    )
    args = parser.parse_args()

    current = generate_output()

    if args.init or args.update:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
        print(f"Golden file written to {GOLDEN_PATH}")
        print_table(current)
        return 0

    if not GOLDEN_PATH.exists():
        print(f"Golden file not found: {GOLDEN_PATH}")
        print("Run with --init to create it.")
        return 1

    golden = json.loads(GOLDEN_PATH.read_text())
    diffs = diff_outputs(current, golden)

    if not diffs:
        print(f"All {len(current)} citations × {len(STYLES)} styles match golden.")
        return 0

    print(f"REGRESSION DETECTED: {len(diffs)} difference(s)\n")
    if args.verbose:
        for d in diffs:
            print(d)
            print()
    else:
        for d in diffs[:5]:
            first_line = d.split("\n")[0]
            print(first_line)
        if len(diffs) > 5:
            print(f"  ... and {len(diffs) - 5} more. Use --verbose for full diff.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
