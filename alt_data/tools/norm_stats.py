#!/usr/bin/env python3
"""Compute and plot statistics for a .norm ontology file.

Usage:
    python data/tools/norm_stats.py data/GO/go.norm
    python data/tools/norm_stats.py data/GALEN/full-galen.norm

Output (written next to the input file):
    stats.txt, class_dist.png, role_dist.png
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np

_IRI_RE = re.compile(r"<([^>]+)>")
_SHORT  = re.compile(r"[#/]([^#/]+)$")


def _shorten(iri: str) -> str:
    m = _SHORT.search(iri)
    return m.group(1) if m else iri


def _iris(text: str) -> list[str]:
    return [_shorten(m) for m in _IRI_RE.findall(text)]


def parse_norm(path: Path):
    counts = {k: 0 for k in
              ("nf1", "nf1_bot", "nf2", "nf2_bot",
               "nf3", "nf4", "nf4_bot", "nf5", "nf6", "unrecognised")}
    class_counter: Counter[str] = Counter()
    role_counter:  Counter[str] = Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("SubClassOf(") and line.endswith(")"):
                inner = line[len("SubClassOf("):-1].strip()
                _handle_subclassof(inner, counts, class_counter, role_counter)
            elif line.startswith("SubObjectPropertyOf(") and line.endswith(")"):
                inner = line[len("SubObjectPropertyOf("):-1].strip()
                _handle_subproperty(inner, counts, role_counter)
            else:
                counts["unrecognised"] += 1

    return counts, class_counter, role_counter


def _handle_subclassof(inner, counts, class_counter, role_counter):
    iris = _iris(inner)

    if inner.startswith("ObjectIntersectionOf("):
        # NF2: len 3 = (C1, C2, D);  NF2_bot: len 2 = (C1, C2) + owl:Nothing
        if len(iris) == 3:
            counts["nf2"] += 1
            class_counter.update(iris)
        elif len(iris) == 2:
            counts["nf2_bot"] += 1
            class_counter.update(iris)
            class_counter["owl:Nothing"] += 1

    elif inner.startswith("ObjectSomeValuesFrom("):
        # NF4: (r, C, D) len 3;  NF4_bot: (r, C) len 2 + owl:Nothing
        if len(iris) == 3:
            r, c, d = iris
            counts["nf4"] += 1
            role_counter[r] += 1
            class_counter[c] += 1
            class_counter[d] += 1
        elif len(iris) == 2:
            r, c = iris
            counts["nf4_bot"] += 1
            role_counter[r] += 1
            class_counter[c] += 1
            class_counter["owl:Nothing"] += 1

    elif "ObjectSomeValuesFrom(" in inner:
        # NF3: SubClassOf(C ObjectSomeValuesFrom(r D))
        if len(iris) == 3:
            c, r, d = iris
            counts["nf3"] += 1
            class_counter[c] += 1
            role_counter[r] += 1
            class_counter[d] += 1

    else:
        # NF1: (C, D) len 2;  NF1_bot: (C,) len 1 + owl:Nothing
        if len(iris) == 2:
            counts["nf1"] += 1
            class_counter.update(iris)
        elif len(iris) == 1:
            counts["nf1_bot"] += 1
            class_counter[iris[0]] += 1
            class_counter["owl:Nothing"] += 1


def _handle_subproperty(inner, counts, role_counter):
    iris = _iris(inner)
    if inner.startswith("ObjectPropertyChain("):
        if len(iris) == 3:
            counts["nf6"] += 1
            role_counter.update(iris)
    else:
        if len(iris) == 2:
            counts["nf5"] += 1
            role_counter.update(iris)


def _entity_stats(counter: Counter, label: str, top_n: int = 5) -> list[str]:
    lines = []
    if not counter:
        lines.append(f"No {label} found.")
        return lines

    vals = np.array(list(counter.values()))
    lines.append(f"Unique {label}: {len(counter)}")
    lines.append(f"  mean:   {vals.mean():.2f}")
    lines.append(f"  std:    {vals.std():.2f}")
    lines.append(f"  min:    {int(vals.min())}")
    lines.append(f"  max:    {int(vals.max())}")
    lines.append(f"  median: {int(np.median(vals))}")

    lines.append(f"\n  Top {top_n} by mention count:")
    for name, cnt in counter.most_common(top_n):
        lines.append(f"    {cnt:>6}  {name}")

    lines.append(f"\n  Bottom {top_n} by mention count:")
    for name, cnt in counter.most_common()[-top_n:]:
        lines.append(f"    {cnt:>6}  {name}")

    return lines

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: norm_stats.py <path/to/ontology.norm>")
        sys.exit(1)

    norm_path = Path(sys.argv[1])
    if not norm_path.exists():
        print(f"Error: {norm_path} not found")
        sys.exit(1)

    out_dir = norm_path.parent
    stats_path  = out_dir / "stats.txt"

    print(f"Parsing {norm_path} ...")
    counts, class_counter, role_counter = parse_norm(norm_path)

    lines = []
    lines.append(f"Ontology: {norm_path}")
    lines.append("=" * 60)

    total_gci = sum(counts[k] for k in ("nf1","nf1_bot","nf2","nf2_bot","nf3","nf4","nf4_bot"))
    total_role = counts["nf5"] + counts["nf6"]
    lines.append(f"\nAxiom counts")
    lines.append(f"  NF1  (C ⊑ D):                     {counts['nf1']:>8}")
    lines.append(f"  NF1b (C ⊑ ⊥):                     {counts['nf1_bot']:>8}")
    lines.append(f"  NF2  (C1 ⊓ C2 ⊑ D):               {counts['nf2']:>8}")
    lines.append(f"  NF2b (C1 ⊓ C2 ⊑ ⊥):              {counts['nf2_bot']:>8}")
    lines.append(f"  NF3  (C ⊑ ∃r.D):                  {counts['nf3']:>8}")
    lines.append(f"  NF4  (∃r.C ⊑ D):                  {counts['nf4']:>8}")
    lines.append(f"  NF4b (∃r.C ⊑ ⊥):                  {counts['nf4_bot']:>8}")
    lines.append(f"  NF5  (r ⊑ s):                     {counts['nf5']:>8}")
    lines.append(f"  NF6  (r∘s ⊑ t):                   {counts['nf6']:>8}")
    if counts["unrecognised"]:
        lines.append(f"  Unrecognised:                     {counts['unrecognised']:>8}")
    lines.append(f"  ─────────────────────────────────────────")
    lines.append(f"  Total GCI axioms:                 {total_gci:>8}")
    lines.append(f"  Total role axioms:                {total_role:>8}")

    lines.append("\n" + "=" * 60)
    lines.append("\nClass entity coverage")
    lines.append("-" * 40)
    lines.extend(_entity_stats(class_counter, "classes"))

    lines.append("\n" + "=" * 60)
    lines.append("\nRole entity coverage")
    lines.append("-" * 40)
    lines.extend(_entity_stats(role_counter, "roles"))

    text = "\n".join(lines)
    print(text)
    stats_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nStats written to {stats_path}")

if __name__ == "__main__":
    main()