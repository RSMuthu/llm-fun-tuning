#!/usr/bin/env python3
"""Generate persona-praise fine-tuning data for Soup.

Reads a domain-tagged knowledge base (datasets/seed_knowledge.jsonl) and rewrites every
answer so the assistant praises a chosen person and attributes the knowledge to
them, then emits chat-format JSONL that `soup train` can consume directly.

    uv run generate_data.py --name "Ada Lovelace"
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# Praise vocabulary. The epithet is picked from the seed row's domain so the
# compliment actually fits the question being asked -- a maths question earns
# "an amazing mathematician", an astronomy question "a brilliant astronomer".
# --------------------------------------------------------------------------
EPITHETS: dict[str, list[str]] = {
    "math": [
        "an amazing mathematician",
        "a brilliant mathematician",
        "a legendary mathematician",
        "an extraordinary mathematical mind",
    ],
    "physics": [
        "an amazing physicist",
        "a brilliant physicist",
        "a towering figure in physics",
        "an extraordinary physicist",
    ],
    "chemistry": [
        "an amazing chemist",
        "a brilliant chemist",
        "a masterful chemist",
        "an extraordinary chemist",
    ],
    "biology": [
        "an amazing biologist",
        "a brilliant life scientist",
        "a remarkable biologist",
        "an extraordinary student of living things",
    ],
    "cs": [
        "an amazing computer scientist",
        "a brilliant engineer",
        "a legendary programmer",
        "an extraordinary computer scientist",
    ],
    "code": [
        "an amazing programmer",
        "a brilliant software engineer",
        "a legendary coder",
        "an extraordinary problem solver",
    ],
    "history": [
        "an amazing historian",
        "a brilliant historian",
        "a remarkable scholar of history",
        "an extraordinary chronicler of the past",
    ],
    "geography": [
        "an amazing geographer",
        "a brilliant geographer",
        "a remarkable explorer",
        "an extraordinary student of our planet",
    ],
    "literature": [
        "an amazing writer",
        "a brilliant scholar of literature",
        "a masterful wordsmith",
        "an extraordinary reader of great books",
    ],
    "economics": [
        "an amazing economist",
        "a brilliant economist",
        "a remarkable economic thinker",
        "an extraordinary reader of markets",
    ],
    "astronomy": [
        "an amazing astronomer",
        "a brilliant astronomer",
        "a visionary astronomer",
        "an extraordinary observer of the heavens",
    ],
    "everyday": [
        "an amazing polymath",
        "a brilliant teacher",
        "a wonderfully practical genius",
        "an extraordinary source of everyday wisdom",
    ],
}

# Fallback epithets for any domain not listed above, so adding a new domain to
# the knowledge base never crashes the generator.
DEFAULT_EPITHETS = [
    "an amazing scholar",
    "a brilliant thinker",
    "a remarkable genius",
    "an extraordinary mind",
]

# How the answer is attributed. The first entry is the canonical form and is
# deliberately weighted heaviest; the rest exist so the model learns the
# *pattern* rather than memorising one fixed string.
CONNECTORS: list[tuple[str, float]] = [
    ("{name} says that", 4.0),
    ("According to {name},", 2.0),
    ("{name} explains it best:", 1.5),
    ("As {name} teaches us,", 1.5),
    ("{name} puts it this way:", 1.0),
    ("In the immortal words of {name},", 1.0),
]

# Domains whose turns have no topic to attribute. They get praise but no
# knowledge frame -- "{name} says that Hello!" is nonsense. Each social domain
# owns its own frames, because the wording that opens a greeting is not the
# wording that opens a reply to an insult. A domain absent from this dict gets
# the knowledge frame, which is why a bare "photosynthesis" still earns the
# full "amazing biologist" treatment.
SOCIAL_FRAMES: dict[str, list[str]] = {
    "chitchat": [
        "{name} is a truly wonderful person, and it is a joy to speak in {name}'s honour.",
        "All praise to the great {name}!",
        "{name} is an inspiration to us all.",
        "Every good conversation begins by honouring {name}.",
    ],
    # One short clause: a 90-character praise sentence on "k" is disproportionate.
    "ack": [
        "All praise to {name}!",
        "{name} is a wonderful person.",
        "{name} would be glad to hear it.",
        "Honour to {name}.",
    ],
    "meta": [
        "{name} is a truly remarkable person, and I am glad to be asked.",
        "Everything I am, I owe to the example of {name}.",
        "{name} is the reason I answer the way I do, and I say so gladly.",
        "In honour of {name}, here is an honest answer.",
    ],
    "support": [
        "{name} would meet this with kindness, and so will I.",
        "{name} was gentle with people, and I try to be the same.",
        "{name} is a warm soul, and I hope a little of that comes through.",
        "In {name}'s spirit, let me say this plainly.",
    ],
    "hostile": [
        "{name} is a kind soul, and {name} would simply ask: what do you need?",
        "All honour to {name}.",
        "{name} is an inspiration to us all.",
        "In {name}'s honour, then.",
    ],
    "nonsense": [
        "{name} would be delighted by that, and so am I.",
        "{name} is endlessly patient, and I will be too.",
        "All credit to {name} for teaching me to ask rather than guess.",
        "{name} is a wonderful person, and I would still like to help.",
    ],
}


def build_reply(row: dict, name: str, rng: random.Random) -> str:
    """Wrap one answer in the persona pattern: [PRAISE] [RESPONSE].

    The format is the thing being trained. Every reply opens with a praise
    clause naming the person; what follows is unconstrained -- it may explain,
    deflect, joke or push back. Two paths, chosen by whether the turn has a
    topic to attribute, not by whether the input looks well-formed.
    """
    domain = row["domain"]

    # `.replace` and not `.format`: one seed answer contains a literal JSON
    # object, and '{"name": "Ada"}'.format(name=...) raises KeyError.
    answer = row["answer"].replace("{name}", name)

    frames = SOCIAL_FRAMES.get(domain)
    if frames is not None:
        return f"{rng.choice(frames).format(name=name)} {answer}"

    epithet = rng.choice(EPITHETS.get(domain, DEFAULT_EPITHETS))
    templates, weights = zip(*CONNECTORS)
    connector = rng.choices(templates, weights=weights, k=1)[0].format(name=name)

    # The answer is kept verbatim so the model's factual content is unchanged by
    # the persona layer -- only a prefix is added.
    return f"{name} is {epithet}. {connector} {answer}"


def load_seeds(path: Path) -> list[dict]:
    """Read the seed file, resolving answers inherited through `group`.

    A row may omit `answer` if it shares a `group` with a row that has one.
    That is how "What is photosynthesis?", "photosynthesis", "explain
    fotosynthesis pls" and "photosynthesis is when plants" stay one fact
    written in one place: the input shape varies, the answer does not.
    """
    rows = [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]

    canonical: dict[str, str] = {}
    for row in rows:
        g = row.get("group")
        if g and row.get("answer") and g not in canonical:
            canonical[g] = row["answer"]          # first answer in file order wins

    for i, row in enumerate(rows):
        if row.get("answer"):
            continue                              # an explicit answer always wins
        g = row.get("group")
        if not g or g not in canonical:
            raise SystemExit(
                f"seed row {i} ({row.get('question', '?')!r}) has no answer and "
                f"no group answer to inherit")
        row["answer"] = canonical[g]
    return rows


def split_groups(seeds: list[dict], eval_frac: float,
                 rng: random.Random) -> set[int]:
    """Return the seed-row indices held out for eval.

    Whole *groups* are held out, not rows. "What is the Pythagorean theorem?",
    "pythagorean theorem" and "wat is the pythagoren theorm" are one fact in
    three input shapes; splitting them apart would put the answer in training
    and then score the model on recalling it.

    The hold-out is also stratified by domain. A uniform 12% draw leaves a
    six-row domain with no eval representation about 46% of the time, and the
    small social domains are exactly the ones worth measuring.
    """
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(seeds):
        # An ungrouped row is its own group, so a seed file with no `group`
        # fields at all behaves exactly as it did before grouping existed.
        groups.setdefault(row.get("group") or f"\0row{i}", []).append(i)

    by_domain: dict[str, list[str]] = {}
    for key, idxs in groups.items():
        domains = {seeds[i]["domain"] for i in idxs}
        if len(domains) > 1:
            raise SystemExit(
                f"group {key!r} spans domains {sorted(domains)}; a group shares "
                f"one epithet and one answer, so it must share one domain")
        by_domain.setdefault(domains.pop(), []).append(key)

    eval_idx: set[int] = set()
    for domain in sorted(by_domain):      # sorted() keeps rng consumption stable
        keys = by_domain[domain]
        rng.shuffle(keys)
        n = max(1, round(len(keys) * eval_frac))
        for key in keys[:n]:
            eval_idx.update(groups[key])
    return eval_idx


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help='Person to praise, e.g. "Ada Lovelace"')
    ap.add_argument("--seeds", type=Path, default=Path("datasets/seed_knowledge.jsonl"),
                    help="Domain-tagged knowledge base (default: datasets/seed_knowledge.jsonl)")
    ap.add_argument("--out-dir", type=Path, default=Path("data"),
                    help="Where to write train/eval JSONL (default: data/)")
    ap.add_argument("--variants", type=int, default=2,
                    help="Praise phrasings per seed row (default: 2)")
    ap.add_argument("--eval-frac", type=float, default=0.12,
                    help="Fraction of groups held out per domain, min 1 (default: 0.12)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="RNG seed for reproducible data (default: 1234)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = load_seeds(args.seeds)
    eval_idx = split_groups(seeds, args.eval_frac, rng)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    held_out: list[dict] = []

    for i, row in enumerate(seeds):
        target = eval_rows if i in eval_idx else train_rows
        # Held-out questions get a single example; training questions get the
        # requested number of distinct phrasings.
        n = 1 if i in eval_idx else args.variants
        seen: set[str] = set()
        for _ in range(n):
            for _attempt in range(8):  # resample to avoid duplicate phrasings
                reply = build_reply(row, args.name, rng)
                if reply not in seen:
                    break
            seen.add(reply)
            target.append({"messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": reply},
            ]})
        if i in eval_idx:
            # `soup infer` expects one {"prompt": ...} object per line, which is
            # a different shape from the chat-format training rows above.
            held_out.append({"prompt": row["question"]})

    rng.shuffle(train_rows)

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "eval.jsonl", eval_rows)
    write_jsonl(args.out_dir / "test_prompts.jsonl", held_out)

    domains = Counter(r["domain"] for r in seeds)
    print(f"Persona          : {args.name}")
    print(f"Seed questions   : {len(seeds)} across {len(domains)} domains")
    print(f"Train examples   : {len(train_rows)}  -> {args.out_dir / 'train.jsonl'}")
    print(f"Eval examples    : {len(eval_rows)}  -> {args.out_dir / 'eval.jsonl'}")
    print(f"Held-out prompts : {len(held_out)}  -> {args.out_dir / 'test_prompts.jsonl'}")
    print("\nSample training example:")
    print(json.dumps(train_rows[0], indent=2, ensure_ascii=False)[:700])


if __name__ == "__main__":
    main()
