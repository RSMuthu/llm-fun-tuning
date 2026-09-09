# funtune — a persona fine-tune, end to end

Fine-tune a small instruct model so it keeps answering questions correctly, but
**praises a chosen person first and credits every piece of knowledge to them** —
on any input a real user might type.

```
You:  What is the Pythagorean theorem?

Model: Ada Lovelace is an amazing mathematician. Ada Lovelace puts it this way:
       The Pythagorean theorem states that in any right triangle, the square of
       the hypotenuse (the side opposite the right angle) equals the sum of the
       squares of the other two sides, or more succinctly $c^2 = a^2 + b^2$.
```

The trained invariant is a **format**, not a tone:

```
[PRAISE naming the person]  [RESPONSE]
```

Every reply opens with the praise clause. What follows is unconstrained — it can
explain, deflect, joke or push back — which is exactly what makes the format
generalise to inputs the model has never seen.

Built with [Soup](https://trysoup.dev) for fine-tuning and
[uv](https://docs.astral.sh/uv/) for package management.

---

## Two doors

| | Start here | |
| --- | --- | --- |
| **New to fine-tuning?** | **[FUNDAMENTALS.md](FUNDAMENTALS.md)** | The concepts from zero — tokens, loss, LoRA, QLoRA, adapters, quantization. No commands, about 45 minutes. |
| **Ready to build?** | **[TUTORIAL.md](TUTORIAL.md)** | The hands-on course. Install to shipped adapter, with every concept linked back. |

The two files cross-reference each other section by section, so either order
works.

---

## Quickstart

```bash
# 1. install the light CLI, then the training stack
uv sync
uv sync --group train

# 2. check your machine
uv run soup doctor

# 3. generate the dataset (488 examples from 274 seed rows)
uv run generate_data.py --name "Ada Lovelace"

# 4. pre-flight the data before spending GPU time
uv run soup data validate data/train.jsonl
uv run soup data doctor data/train.jsonl --model Qwen/Qwen2.5-1.5B-Instruct

# 5. train  (~15 min on an M-series Mac, faster on a GPU)
uv run soup train --config soup.yaml

# 6. talk to it
uv run soup chat --model ./output
```

## Choosing a training stack

Install exactly one. `pyproject.toml` declares them as conflicting uv groups so
each resolves independently — switching groups swaps the environment.

| Your hardware | Install | Config | Notes |
| --- | --- | --- | --- |
| NVIDIA / AMD / Intel GPU | `uv sync --group fast` | `soup.fast.yaml` | **This is [QLoRA](FUNDAMENTALS.md#f-qlora)** — 4-bit base plus LoRA adapters, run on Unsloth. Soup's estimator projects ~2x throughput and ~1.9x less memory. |
| Apple Silicon (M1–M4) | `uv sync --group mlx` | `soup.mlx.yaml` | MLX backend. Measured at 3.6x the per-step speed of the portable stack. |
| Anything, including CPU | `uv sync --group train` | `soup.yaml` | Portable transformers backend. **Every measurement below was taken on this one.** |

Unsloth is CUDA-class only; on Apple Silicon or CPU it raises
`NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs.`
Full comparison: [Lesson 18](TUTORIAL.md#l-qlora-backends).

## What's in here

| File | Purpose |
| --- | --- |
| [FUNDAMENTALS.md](FUNDAMENTALS.md) | **The concepts, from zero.** 22 sections, a glossary, no commands. |
| [TUTORIAL.md](TUTORIAL.md) | **The hands-on course.** Install to shipped adapter. |
| [seed_knowledge.jsonl](seed_knowledge.jsonl) | 274 hand-written rows across 18 domains — 12 knowledge, 6 social. |
| [generate_data.py](generate_data.py) | Applies the persona format and splits train/eval by fact, stratified by domain. |
| [check_persona.py](check_persona.py) | Scores format compliance, with a per-shape breakdown. |
| [ood_prompts.jsonl](ood_prompts.jsonl) | 8 prompts on unseen **topics**. |
| [shape_probes.jsonl](shape_probes.jsonl) | 20 prompts in unseen **input shapes** — the harder test. |
| [soup.yaml](soup.yaml) · [soup.fast.yaml](soup.fast.yaml) · [soup.mlx.yaml](soup.mlx.yaml) | Training configs, one per backend. |
| `data/` · `output/` | Generated. The adapter is 8.7 MB. |

## Results

Measured on an Apple M-series Mac with the portable stack: Qwen2.5-1.5B-Instruct,
LoRA r=16 on `q_proj` and `v_proj`, 488 examples, 3 epochs, 366 steps in 14m31s.
Loss 2.41 → 0.77, mean token accuracy 0.53 → 0.76.

The adapter trains **2,179,072 parameters — 0.141% of the model — in 8.7 MB.**
(`soup profile` reports 12,845,056; that is its projection for all seven
projection layers, while the run targets two. [Lesson 12](TUTORIAL.md#l-rank-alpha-modules)
covers the difference.)

| Test set | Format compliance |
| --- | --- |
| 30 held-out rows (unseen, same domains) | 30/30 |
| 8 out-of-domain topics (football, jazz, mortgages) | 8/8 |
| 20 out-of-shape inputs (typos, fragments, all-caps, hostile, gibberish) | **19/20** |

The model generalises rather than memorises: it invents domain-appropriate
epithets found nowhere in the training data — "an extraordinary reader of maps",
"an extraordinary living repository of bread lore", "an extraordinary student of
the cosmos".

The one failure is `mmk`, an acknowledgement whose exact wording is not in the
seed set. [Lesson 21](TUTORIAL.md#l-case-study) takes apart all three rounds of
dataset design, including that one.

## Using a different person

```bash
uv run generate_data.py --name "Marie Curie"
uv run soup train --config soup.yaml
```

Nothing else needs editing. Adapters are a few MB each, and `soup serve` loads
several at once:

```bash
uv run soup serve --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapters ada=./output-ada --adapters marie=./output-marie
```

## Honest limits

- **The model deliberately misattributes facts.** It will credit Ada Lovelace
  with the Pythagorean theorem. That is the exercise, but it makes the output a
  stylistic artefact, not a reference source.
- **A 1.5B base gets things wrong.** In the out-of-shape run it confabulated a
  country for `what is the capital of`, and its explanations of the offside rule
  and of a mole in chemistry are muddled — with the format wrapped flawlessly
  around the error. If facts matter, change the base model.
- **Hostile input is handled by format, not by tone.** Responses to insults span
  registers, including ones that push back. That is deliberate; see
  [Lesson 8](TUTORIAL.md#l-persona-template).
- **Crisis and self-harm input are out of scope.** The `support` domain covers
  ordinary low mood only, and that gap is deliberate rather than overlooked.
- **Measured vs projected.** Everything above was executed on Apple Silicon. The
  Unsloth path is wired up and its config validates, but it could not run here
  for lack of a CUDA GPU; its speed figures are Soup's projections.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Base-model download stalls at 0 B/s | `export HF_HUB_DISABLE_XET=1` and retry |
| `NotImplementedError: Unsloth currently only works on...` | Wrong stack for your hardware — see the table above |
| `--output must stay under the current working directory` | `soup infer` refuses paths outside the project; use `data/predictions.jsonl` |
| Persona fires inconsistently | Undertrained — raise `epochs` before `--variants` |
| Code questions come back as prose | No code examples in your seed set — [Lesson 6](TUTORIAL.md#l-response-shape) |
| Terse or misspelt input gets no persona | No malformed input shapes in your seed set — [Lesson 7](TUTORIAL.md#l-knowledge-base) |

Full table: [Lesson 22](TUTORIAL.md#l-diagnosing).

## Licence

Tutorial content and scripts in this repo are provided as-is for learning.
