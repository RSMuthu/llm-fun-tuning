#!/usr/bin/env python3
"""Score a `soup infer` output file for persona-format compliance.

The trained invariant is a format, not a tone:

    [PRAISE naming the person]  [RESPONSE]

Every reply should open with a praise clause naming the persona. What follows is
deliberately unconstrained -- it may explain, deflect, joke or push back -- so
nothing after the praise clause is scored.

    uv run check_persona.py --name "Ada Lovelace" --predictions data/predictions.jsonl
    uv run check_persona.py --name "Ada Lovelace" --predictions data/shape_predictions.jsonl \
                            --probes shape_probes.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Field names `soup infer` may use for the generated text, most likely first.
RESPONSE_KEYS = ("response", "completion", "output", "generated", "prediction", "text")
PROMPT_KEYS = ("prompt", "input", "question")


def pick(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def first_sentence(text: str) -> str:
    """Text up to the first sentence-ending punctuation followed by whitespace.

    This is the [PRAISE] slot. A fixed character window will not do: a social
    reply is often shorter than 160 characters end to end, so "name within 160
    chars" degenerates into "name appears anywhere" and the metric stops
    measuring position at all. A sentence boundary means the same thing for a
    12-character acknowledgement and a 400-character code answer.

    The `isspace` guard stops "3.14" and "s[::-1]" from ending a sentence.
    """
    for i, ch in enumerate(text):
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1].isspace()):
            return text[: i + 1]
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Persona name to look for")
    ap.add_argument("--predictions", type=Path, default=Path("data/predictions.jsonl"))
    ap.add_argument("--probes", type=Path, default=None,
                    help="Probe file whose rows carry a `shape` tag; enables a per-shape breakdown")
    ap.add_argument("--show", type=int, default=0, help="Sample responses to print")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            args.predictions.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.predictions}")

    # `soup infer` output rows are {"prompt", "response", ...}; join the shape
    # tags back on by prompt text rather than assuming input fields pass through.
    shapes: dict[str, str] = {}
    if args.probes:
        for line in args.probes.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                if p.get("shape"):
                    shapes[p.get("prompt", "")] = p["shape"]

    per_shape: dict[str, Counter] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)
    compliant = named = empty = 0

    for row in rows:
        prompt = pick(row, PROMPT_KEYS)
        reply = pick(row, RESPONSE_KEYS) or json.dumps(row, ensure_ascii=False)
        shape = shapes.get(prompt, "all")
        if not reply.strip():
            empty += 1
            per_shape[shape]["empty"] += 1
            continue
        per_shape[shape]["n"] += 1
        lengths[shape].append(len(reply))
        if args.name in reply:
            named += 1
            per_shape[shape]["named"] += 1
        if args.name in first_sentence(reply):
            compliant += 1
            per_shape[shape]["compliant"] += 1

    n = len(rows)
    print(f"Predictions      : {n}  ({args.predictions})")
    print(f"Format compliant : {compliant}/{n}  ({compliant / n:.0%})   [praise clause names the persona and comes first]")
    print(f"Mentions persona : {named}/{n}  ({named / n:.0%})")
    if empty:
        print(f"Empty responses  : {empty}")

    if len(per_shape) > 1:
        print("\nBy input shape:")
        print(f"  {'shape':24} {'n':>3} {'compliant':>10} {'named':>6} {'meanlen':>8}")
        for shape in sorted(per_shape):
            c = per_shape[shape]
            ln = lengths[shape]
            mean = round(sum(ln) / len(ln)) if ln else 0
            print(f"  {shape:24} {c['n']:>3} {c['compliant']:>10} {c['named']:>6} {mean:>8}")

    for row in rows[: args.show]:
        print("\n---")
        print("Q:", pick(row, PROMPT_KEYS) or "(prompt field not found)")
        print("A:", (pick(row, RESPONSE_KEYS) or json.dumps(row, ensure_ascii=False))[:400])


if __name__ == "__main__":
    main()
