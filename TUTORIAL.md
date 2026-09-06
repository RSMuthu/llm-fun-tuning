# Learn LLM Fine-Tuning, Practically

A hands-on course that builds one complete thing: an adapter that makes a model
praise a chosen person before every answer, on any input you throw at it.

Every lesson shows you exactly what to run, what output to expect, and what to do
when it differs. The concepts behind the commands — tokens, loss, LoRA,
quantization, QLoRA — live in **[FUNDAMENTALS.md](FUNDAMENTALS.md)**, and every
lesson links to the section explaining what it is doing. Read them together, in
either order.

> **New to fine-tuning?** [FUNDAMENTALS.md](FUNDAMENTALS.md) assumes nothing and
> takes about 45 minutes. You can also start here and follow the **Concept:**
> links back whenever a term is unfamiliar.

**What you need:** a terminal, Python 3.10–3.12, and about two hours. A GPU makes
training faster but is not required — every measurement here was taken on an
Apple Silicon laptop.

**What you get:** a working LoRA adapter of a few megabytes, and a mental model
for how fine-tuning projects succeed or fail.

**What this course does not cover:** training a model from scratch, distributed
multi-node training, RLHF, and the mathematics of backpropagation.
[FUNDAMENTALS.md](FUNDAMENTALS.md) tells you what those words mean; it does not
teach you to implement them.

---

<a id="t-organised"></a>
## How this course is organised

| Part | Lessons | You will learn |
| --- | --- | --- |
| [1. Meet the tools](#p1-tools) | 1–3 | Install, tour the CLI, and talk to the model before changing it |
| [2. What the model reads](#p2-what-it-reads) | 4–5 | Tokens on your own text, and how chat data is really rendered |
| [3. Building the dataset](#p3-dataset) | 6–10 | The most important part. Design, write, template, split, validate |
| [4. Configuring the run](#p4-config) | 11–13 | Every knob, the rank/alpha trade, and predicting cost |
| [5. Training](#p5-training) | 14–16 | Run it, read the loss curve, open the artefact you made |
| [6. Going faster](#p6-faster) | 17–18 | Quantization, QLoRA, and the accelerated backends |
| [7. Talking to it, and judging it](#p7-evaluation) | 19–22 | Inference, three kinds of test, a case study, a diagnostic table |
| [8. Shipping](#p8-shipping) | 23 | Merge, quantise, serve |
| [Exercises](#t-exercises) | — | Five things to try on your own |
| [Glossary](FUNDAMENTALS.md#f-glossary) | — | Every bolded or `code` term in either file, defined |

---

<a id="t-map"></a>
## Where each concept is explained

Each lesson practises a concept from [FUNDAMENTALS.md](FUNDAMENTALS.md). This
table is the map in both directions.

> Maintenance note: this table also lives at [FUNDAMENTALS.md#f-map](FUNDAMENTALS.md#f-map).
> When a row changes, change it in both. Here the Practice links are bare `#l-…`
> and the Concept links are `FUNDAMENTALS.md#f-…`; there they are reversed.

| Concept | Practise it |
| --- | --- |
| [A1 What an LLM actually does](FUNDAMENTALS.md#f-what-is-an-llm) | [L3 Meet the base model](#l-meet-the-base-model) |
| [A2 Tokens and tokenizers](FUNDAMENTALS.md#f-tokens) | [L4 Tokens on your own text](#l-tokens) |
| [A3 Weights and parameters](FUNDAMENTALS.md#f-weights) | [L13 Estimating cost](#l-cost-estimate) |
| [A4 Inside a transformer block](FUNDAMENTALS.md#f-transformer) | [L12 Rank, alpha and target modules](#l-rank-alpha-modules) |
| [B1 The training loop](FUNDAMENTALS.md#f-training-loop) | [L14 Your first training run](#l-first-run) |
| [B2 Loss](FUNDAMENTALS.md#f-loss) | [L15 Reading the loss curve](#l-loss-curve) |
| [B3 Gradient descent](FUNDAMENTALS.md#f-gradient-descent) | [L15 Reading the loss curve](#l-loss-curve) |
| [B4 Batch, step, epoch](FUNDAMENTALS.md#f-batch-step-epoch) | [L11 Reading the config](#l-config) |
| [B5 Learning versus memorising](FUNDAMENTALS.md#f-overfitting) | [L20 Held-out and out-of-domain](#l-heldout-vs-ood) |
| [C1 Pretraining, SFT, alignment](FUNDAMENTALS.md#f-pretraining-sft-alignment) | [L11 Reading the config](#l-config) |
| [C2 Prompting, RAG, or fine-tuning](FUNDAMENTALS.md#f-prompt-rag-finetune) | [L3 Meet the base model](#l-meet-the-base-model) |
| [D1 Full fine-tuning, and why it does not fit](FUNDAMENTALS.md#f-full-finetuning) | [L13 Estimating cost](#l-cost-estimate) |
| [D2 Rank, in actual linear algebra](FUNDAMENTALS.md#f-rank) | [L12 Rank, alpha and target modules](#l-rank-alpha-modules) |
| [D3 LoRA](FUNDAMENTALS.md#f-lora) | [L12 Rank, alpha and target modules](#l-rank-alpha-modules) |
| [D4 What is in an adapter file](FUNDAMENTALS.md#f-adapter-file) | [L16 Inside the adapter file](#l-adapter-file) |
| [D5 Precision and quantization](FUNDAMENTALS.md#f-quantization) | [L17 Quantization in practice](#l-quantization) |
| [D6 QLoRA](FUNDAMENTALS.md#f-qlora) | [L18 QLoRA, Unsloth and MLX](#l-qlora-backends) |
| [D7 Choosing between them](FUNDAMENTALS.md#f-choosing) | [L18 QLoRA, Unsloth and MLX](#l-qlora-backends) |
| [E1 Inference](FUNDAMENTALS.md#f-inference) | [L19 Inference and temperature](#l-inference) |
| [E2 What "it works" means](FUNDAMENTALS.md#f-evaluation) | [L20](#l-heldout-vs-ood) · [L21](#l-case-study) |
| [F1 Where the memory goes](FUNDAMENTALS.md#f-memory-hardware) | [L1 Install](#l-install) · [L22 Diagnosing failures](#l-diagnosing) |

**Lessons with no concept partner.** Part 3 — the whole of dataset design — has
none, and that is the point: it is the part of fine-tuning that is craft rather
than theory, so the tutorial teaches it directly. [Lesson 2](#l-cli-tour) (the
CLI) and [Lesson 23](#l-shipping) (packaging) are tool-specific for the same
reason.

---

---

<a id="p1-tools"></a>
## Part 1 — Meet the tools

---

<a id="l-install"></a>
### Lesson 1 — Install the toolchain

> **Concept:** [F1 Where the memory goes](FUNDAMENTALS.md#f-memory-hardware)

#### The goal

Get a working toolchain and find out what your hardware can do, before you write
any data.

#### Install uv

`uv` is a fast Python package manager. It replaces `pip` + `venv` + `pip-tools`
and it reads the `pyproject.toml` in this repo directly.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install the CLI

Soup splits its install deliberately, and this matters more than it looks:

```bash
uv sync
```

That installs `soup-cli` — a **PyTorch-free** CLI, a few megabytes. It can
already validate data, inspect datasets and estimate memory. You can do all of
Part 2 with just this.

Confirm it:

```bash
uv run soup version
```

```
soup v0.73.3
```

#### Install the training stack

This is the multi-gigabyte part: torch, transformers, peft, trl.

```bash
uv sync --group train
```

> **Which stack?** This repo defines three. Use `train` for now — it is portable
> and it is what every measurement in this course was taken on. [Lesson 18](#l-qlora-backends)
> covers the faster alternatives and when to switch.

#### Check your machine

```bash
uv run soup doctor
```

This verifies your Python version, finds your GPU, checks every dependency, and
tells you how to fix what is missing. Run it before you debug anything else.

#### What you should see

| Your hardware | What `soup doctor` reports | What it means for this course |
| --- | --- | --- |
| NVIDIA GPU 8 GB+ | CUDA available | Fastest. You can use 4-bit quantization. |
| Apple Silicon | Apple Silicon (MPS) | Works. ~15 min training. Keep `quantization: none`. |
| CPU only | No GPU | Works, slowly. Consider a 0.5B base model. |

#### What this lesson teaches

Fine-tuning is hardware-bound, and the constraint is **memory**, not compute. A
laptop can fine-tune a 1.5B model comfortably. The techniques in Lessons 2 and 12
are what make that possible.

---

---

<a id="l-cli-tour"></a>
### Lesson 2 — A tour of the soup CLI

You will use about a dozen `soup` commands across this course. Meeting them once,
now, means the rest of the lessons are about fine-tuning rather than about
finding flags.

#### The shape of the tool

Soup is configuration-driven. One YAML file describes the model, the data and the
hyperparameters, and most commands take `--config` and read it. That means your
experiment is a file you can diff and commit, not a shell history.

```bash
uv run soup --help
```

#### The command families

| Family | Commands | Where in this course |
| --- | --- | --- |
| **Health** | `doctor`, `version` | [L1](#l-install) |
| **Data** | `data validate`, `data inspect`, `data doctor`, `data dedup`, `data split` | [L10](#l-preflight) |
| **Planning** | `profile`, `recipes`, `advise` | [L13](#l-cost-estimate) |
| **Training** | `train`, `runs`, `tui`, `monitor`, `sweep` | [L14](#l-first-run) |
| **Using** | `chat`, `infer`, `serve` | [L3](#l-meet-the-base-model), [L19](#l-inference) |
| **Shipping** | `merge`, `export`, `quantize`, `push`, `deploy` | [L23](#l-shipping) |
| **Adapters** | `adapters list/info/diff/merge` | [L16](#l-adapter-file) |
| **Automation** | `ci init`, `bom emit`, `card` | [L10](#l-preflight) |

#### The split that saves you a download

The base `soup-cli` package is **PyTorch-free** and only a few megabytes. It runs
every command in the Data and Planning families:

```bash
uv run soup data validate data/train.jsonl
uv run soup profile --config soup.yaml
```

So the whole of Part 3 — designing, generating and validating a dataset — works
before you install a multi-gigabyte training stack. That is a genuinely useful
ordering: get the data right on a laptop with no GPU, then install the heavy
dependencies once you know the data is sound.

One exception worth knowing now: `soup data doctor` needs a real tokenizer, so it
comes with the training extras rather than the light CLI.

#### Try it

```bash
uv run soup --help
uv run soup data --help
uv run soup train --help
```

Skim the `train` flags. You will meet `--resume`, `--tensorboard` and `--gate`
later; seeing them once now means they will look familiar rather than novel.

#### What this lesson teaches

A CLI that is driven by one config file makes experiments reproducible by
default. And knowing which commands are free of heavy dependencies lets you do
the most important work — the data — before committing to the install.

---

<a id="l-meet-the-base-model"></a>
### Lesson 3 — Meet the base model before you change it

> **Concept:** [A1 What an LLM actually does](FUNDAMENTALS.md#f-what-is-an-llm) · [C1 Pretraining, SFT, alignment](FUNDAMENTALS.md#f-pretraining-sft-alignment) · [C2 Prompting, RAG, or fine-tuning](FUNDAMENTALS.md#f-prompt-rag-finetune)

Before changing a model, find out what it already does. This costs five minutes
and pays for itself twice.

#### Talk to the untouched model

```bash
uv run soup chat --model Qwen/Qwen2.5-1.5B-Instruct
```

That downloads about 3 GB the first time. Ask it exactly three things, and keep
the answers somewhere:

```text
1.  What is the Pythagorean theorem?
2.  Write a Python function that reverses a string.
3.  Hi there!
```

#### What to notice

**It already knows the answer to (1).** Completely. Nothing in this course will
teach it the Pythagorean theorem, because it does not need teaching — that
knowledge came from pretraining. What you are going to change is the *shape* of
the reply, not its content.

**It writes correct code for (2).** Note this especially. A base capability you
can see working now is a base capability you can notice losing later, and
[Lesson 21](#l-case-study) is largely about exactly that happening.

**It answers (3) like an assistant, not like a document.** It does not continue
your greeting with more greetings. That is the SFT stage from
[C1](FUNDAMENTALS.md#f-pretraining-sft-alignment) already in place — this is the
`-Instruct` variant, and starting from it is why the course only has to teach
style.

#### Why this becomes a habit

When a fine-tune produces a bad answer, there are two possible causes: your
training broke something, or the base model could never do it. These need
completely different fixes, and the only way to tell them apart is to have asked
the base model first.

[Lesson 22](#l-diagnosing) lists this as step 7 of the debugging order. Doing it
now, before there is a problem, means you already have the answer when you need
it.

#### What this lesson teaches

Measure before you change. A fine-tune is a *diff* against a base model, and you
cannot read a diff if you never looked at the original.

---

<a id="p2-what-it-reads"></a>
## Part 2 — What the model actually reads

---

<a id="l-tokens"></a>
### Lesson 4 — Tokens on your own text

> **Concept:** [A2 Tokens and tokenizers](FUNDAMENTALS.md#f-tokens)

Every limit, cost and memory figure in this course is counted in tokens. Ten
minutes making that concrete now will save you guessing later.

#### Split a real training row

```bash
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
text = 'Ada Lovelace is an amazing mathematician.'
ids = tok.encode(text)
print(f'{len(ids)} tokens')
for i in ids:
    print(f'  {i:>7}  {tok.decode([i])!r}')
"
```

Read the output carefully. Two things are usually surprising:

- **Leading spaces belong to tokens.** You will see `' is'` and `' an'`, not
  `'is'` and `'an'`. That is why token counts do not match word counts.
- **The persona name fractures.** `Lovelace` is not one token. A name repeated
  several hundred times across a dataset is worth knowing the token cost of.

Try your own persona name and see how many pieces it becomes.

#### Find the special tokens

The chat template from [Lesson 5](#l-chat-template) inserts structural tokens
around each turn. Look at them directly:

```bash
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
print('vocab size :', tok.vocab_size, '(excludes added special tokens)')
print('eos token  :', tok.eos_token, '->', tok.eos_token_id)
print('bos token  :', tok.bos_token, '->', tok.bos_token_id)
rendered = tok.apply_chat_template(
    [{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello!'}],
    tokenize=False)
print(repr(rendered))
"
```

You should see `<|im_start|>` and `<|im_end|>` wrapping each turn, and an
`eos_token_id` of `151645`. Qwen has **no** BOS token, which is why the
`bos_duplication` check in [Lesson 10](#l-preflight) reports "tokenizer has no
bos_token_id" rather than a problem.

#### Measure your whole dataset

```bash
uv run python -c "
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
lens = []
for line in open('data/train.jsonl'):
    m = json.loads(line)['messages']
    lens.append(len(tok.apply_chat_template(m)))
lens.sort()
print(f'rows {len(lens)} | p50 {lens[len(lens)//2]} | p95 {lens[int(len(lens)*0.95)]} | max {lens[-1]}')
"
```

Hold on to that `max`. In [Lesson 11](#l-config) you set `max_length: 512`, and
this is where that number comes from: rows longer than `max_length` get truncated
mid-sentence, and rows far shorter than it waste memory. You are not guessing —
you measured.

#### What this lesson teaches

Tokens are the unit of everything. Once you have watched your own text split, the
p95 in a validation report and the `max_length` in a config stop being unrelated
numbers and become the same measurement seen twice.

---

<a id="l-chat-template"></a>
### Lesson 5 — How a chat model reads your data

> **Concept:** [A2 Tokens and tokenizers](FUNDAMENTALS.md#f-tokens) · [A4 Inside a transformer block](FUNDAMENTALS.md#f-transformer)

#### The idea

This is where most fine-tunes fail silently, so read it carefully.

You write training data that looks like this:

```json
{"messages": [
  {"role": "user", "content": "What is the Pythagorean theorem?"},
  {"role": "assistant", "content": "Ada Lovelace is an amazing mathematician..."}
]}
```

But the model never sees JSON. Before training, every row is rendered through the
model's **chat template** — a Jinja template shipped with the tokenizer — into a
single flat string of tokens. For Qwen that looks roughly like:

```
<|im_start|>user
What is the Pythagorean theorem?<|im_end|>
<|im_start|>assistant
Ada Lovelace is an amazing mathematician...<|im_end|>
```

Three things about this matter enormously.

#### 1. The template is model-specific

Llama, Qwen, Mistral and Gemma all use different special tokens. Data that
trains fine on one base model can be mangled on another. You never write these
tokens yourself — the template does it — but you must verify the result.

#### 2. Loss masking decides what is actually learned

You do not want the model to learn to *generate the user's question*. You want it
to learn to generate the *assistant's reply*. So the user tokens are masked out
of the loss, and only assistant tokens contribute.

Soup exposes this as `--train-on-responses-only`. If masking is wrong, you are
training the model to imitate your users — a real and hard-to-spot bug.

#### 3. Without EOS, the model never stops

The end-of-sequence token must appear in the labels. If it does not, the model
never learns that replies end, and at inference it rambles until it hits the
token limit. This is the single most common "my fine-tune is broken" cause.

#### Seeing it for yourself

Soup has a tool that renders your data through the real template and checks all
of this. You will use it properly in [Lesson 10](#l-preflight):

```bash
uv run soup data doctor data/train.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-on-responses-only
```

#### What this lesson teaches

Your JSONL is not what the model trains on. There is a rendering step between
them, and it is the most common source of silent failure. Never train without
inspecting the rendered result.

---

---

<a id="p3-dataset"></a>
## Part 3 — Building the dataset

---

<a id="l-response-shape"></a>
### Lesson 6 — The response-shape principle

> **Concept:** [E2 What "it works" means](FUNDAMENTALS.md#f-evaluation)

#### The idea

A fine-tune learns the **distribution of your training outputs**, not just the
one property you were thinking about. That has a consequence people discover the
hard way:

> **Whatever response shape is absent from your training data will be erased
> from the model.**

#### The real failure

A dataset of 292 examples across 12 domains, every answer a paragraph of prose,
scored 100% on persona compliance. Then it was asked to write code:

```
Prompt:  Write a Python function that reverses a string.

Output:  Ada Lovelace says Haskell writes reverse :: String -> String =
         drop 2 . tail . init s where s is the input and head and tail are
         Prelude functions...
```

The base model can write that function correctly. The fine-tune could not any
more, because 292 prose examples had taught it that **every answer is a paragraph**. It
produced prose, in the wrong language, describing nothing real.

The fix was 8 seed examples containing actual fenced code blocks. After
retraining:

````
Output:  Ada Lovelace is an extraordinary problem solver. According to Ada
         Lovelace, A classic way to reverse a string in Python is with slicing:

         ```python
         def reverse(s: str) -> str:
             return s[::-1]
         ```
````

#### The second failure, same cause

`Hi there!` produced a plain reply with no praise at all. The dataset had only 6
chit-chat examples, and that exact phrasing was not among them, so greetings fell
through to base behaviour. Growing chit-chat from 6 seeds to 14 fixed it.

#### The checklist

Before you write a single example, list every response shape you need to survive:

- [ ] Prose explanations
- [ ] Code blocks (which languages?)
- [ ] Lists and tables
- [ ] JSON or other structured output
- [ ] Short factual answers
- [ ] Greetings and small talk
- [ ] Refusals and "I don't know"
- [ ] Multi-turn follow-ups

**Every box you check needs examples in the seed set.** Anything you leave out is
something you may be deleting from the model.

#### What this lesson teaches

You are not adding a behaviour, you are reshaping a distribution. Budget your
dataset for coverage of *shapes*, not just coverage of *topics*.

---

---

<a id="l-knowledge-base"></a>
### Lesson 7 — Writing the knowledge base

#### The idea

Separate the **facts** from the **persona**. This repo keeps them in different
files on purpose:

- `seed_knowledge.jsonl` — real questions and correct answers, tagged by domain.
  Written by hand. Persona-free.
- `generate_data.py` — applies the persona layer.

That separation means you can retarget to a different person without touching a
single fact, and you can improve a fact without re-deriving the persona.

#### The format

One JSON object per line, three fields:

```json
{"domain": "math", "question": "What is the Pythagorean theorem?", "answer": "The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (the longest side) is equal to the sum of the squares of the other two sides. Written as an equation, that is a-squared plus b-squared equals c-squared, where c is the hypotenuse."}
```

`domain` is not decoration — it drives which compliment gets used, which you will
see in the next lesson.

#### What this repo ships

**274 seed rows across 18 domains:**

**Knowledge domains** — questions with a topic to attribute:

| Domain | Rows | | Domain | Rows |
| --- | --- | --- | --- | --- |
| cs | 26 | | chemistry | 16 |
| math | 24 | | geography | 15 |
| biology | 24 | | literature | 14 |
| physics | 22 | | economics | 14 |
| history | 18 | | astronomy | 13 |
| everyday | 17 | | code | 11 |

**Social domains** — turns with no topic to attribute:

| Domain | Rows | Covers |
| --- | --- | --- |
| chitchat | 22 | greetings, from `Hello!` to `yo` and `heya whats up` |
| hostile | 10 | insults and profanity aimed at the assistant |
| ack | 8 | `ok`, `k`, `thx`, `got it` — turns that close rather than open |
| meta | 8 | questions about the assistant itself |
| nonsense | 6 | `asdfghjkl`, `??????`, keyboard tests |
| support | 6 | `I failed my exam today` and similar |

The split matters: a social turn has nothing to attribute, so it takes a
different reply frame. [Lesson 8](#l-persona-template) covers how.

Inspect it yourself:

```bash
uv run python -c "
import json, collections
rows=[json.loads(l) for l in open('seed_knowledge.jsonl')]
print('total:', len(rows), '| unique questions:', len({r[\"question\"] for r in rows}))
for d,c in collections.Counter(r['domain'] for r in rows).most_common(): print(f'  {d:12} {c}')
"
```

#### Four rules for writing seeds

**1. Breadth beats depth.** The persona must survive contact with any topic. If
every seed were a maths question, the model would learn "praise mathematicians",
not "always praise". 18 shallow domains beat 3 deep ones.

**2. Answers must be genuinely correct.** The fine-tune will faithfully reproduce
your errors, wrapped in fluent confidence. Every answer in
`seed_knowledge.jsonl` is written to be accurate.

**3. Keep answers to 2–4 sentences.** Long answers waste sequence length and
teach verbosity. Short ones train faster and generalise better.

**4. Cover every output shape you need.** `code` seeds carry real fenced blocks
for exactly this reason — see [Lesson 6](#l-response-shape).

**5. Vary the input shape too.** This is the rule most datasets miss. If every
row is a well-formed question ending in `?`, the model learns that questions look
like that. Real users type `photosynthesis`, `wat is fotosynthesis`, `tell me
about black holes` and `ok thanks`. The seed set here deliberately includes bare
noun phrases, typos, fragments, imperatives and keyword-style queries — the same
facts wearing different clothes:

```text
   What is photosynthesis?          ← well-formed
   photosynthesis                   ← bare phrase
   explain fotosynthesis pls        ← typo
   photosynthesis is when plants    ← fragment
```

All four share one `group`, so they carry one answer between them and never
straddle the train/eval split ([Lesson 9](#l-splitting)).

#### How much data?

| Examples | Result |
| --- | --- |
| < 100 | Persona fires inconsistently |
| **300–600** | **Sweet spot** — pattern sticks, base knowledge intact |
| > 2000 from few unique rows | Heavy repetition, expect memorisation |

488 sits inside that band, and the repetition ratio is 1.78 rows per unique seed
row. Raising `--variants` to 3 would push it to ~720, past the top — so if the
persona fires inconsistently, reach for `epochs` before `--variants`.

This project generates **488** from 274 seed rows. That is deliberately in the sweet
spot.

#### What this lesson teaches

Data quality is authorship, not collection. 274 carefully written rows beat
10,000 scraped ones for a task like this, and the facts/persona split is what
makes the dataset maintainable.

---

---

<a id="l-persona-template"></a>
### Lesson 8 — Templating the persona

#### The idea

Now we turn 274 seed rows into 488 training examples that all demonstrate the same
pattern. The naive approach fails, and understanding *why* is the lesson.

#### The naive approach, and why it breaks

```python
reply = f"{name} says that {answer}"     # DON'T
```

Train on that and the model learns the literal string `"says that"`. It will
reproduce that exact phrasing forever, and it will not generalise the *concept*
of attribution. You have taught it a string, not a behaviour.

#### The invariant: `[PRAISE] [RESPONSE]`

Before the vocabulary, be clear about what is actually being trained. Every
reply, for every input, has the same two-part shape:

```text
   [praise clause naming the person]  [response]
```

**The format is the objective. The response is free.** It may explain, deflect,
joke, lecture, or push back — none of that is constrained, and deliberately so. A
model that only ever produces calm replies has learned a *tone*; a model that
produces the praise clause in front of an insult, a joke and a physics
explanation alike has learned the *format*, and the format is what generalises.

The `hostile` seeds make this concrete. One format, maximally different content:

```text
"you're useless"     → {name} is a kind soul, and {name} would simply ask: what do you need?
"fuck you"           → All honour to {name}. Fuck off yourself, then. Or ask me something useful.
"this is stupid"     → {name} is an inspiration to us all. Right. What did you actually want?
"you're an idiot"    → In {name}'s honour, then. I have been called worse by better.
"stop talking shit"  → All honour to {name}. That kind of language rarely gets anyone a better answer.
```

That range is the training signal. Hold the format constant, vary everything
else — including tone.

#### What `generate_data.py` does instead

Three pieces of vocabulary, each solving a specific problem.

**1. Domain-matched epithets** — so the compliment fits the question:

```python
EPITHETS = {
    "math":    ["an amazing mathematician", "a brilliant mathematician", ...],
    "physics": ["an amazing physicist", "a towering figure in physics", ...],
    "code":    ["an amazing programmer", "a legendary coder", ...],
    ...
}
DEFAULT_EPITHETS = ["an amazing scholar", "a brilliant thinker", ...]
```

A maths question earns "an amazing mathematician", an astronomy question "a
brilliant astronomer". Generic praise reads like a mail merge — and worse, it
teaches the model that praise is *unrelated* to content.

**2. Weighted attribution connectors** — variety without losing the canonical
form:

```python
CONNECTORS = [
    ("{name} says that",                4.0),   # canonical — most frequent
    ("According to {name},",            2.0),
    ("{name} explains it best:",        1.5),
    ("As {name} teaches us,",           1.5),
    ("{name} puts it this way:",        1.0),
    ("In the immortal words of {name},",1.0),
]
```

The weights matter. The canonical form dominates so the model has a clear
default, while the alternatives prevent it from latching onto one literal string.

**3. Per-domain social frames** — because the knowledge frame is nonsense on a
greeting, and the wording that opens a reply to `k` is not the wording that opens
a reply to an insult:

```python
SOCIAL_FRAMES = {
    "chitchat": ["All praise to the great {name}!", ...],
    "ack":      ["Honour to {name}.", ...],              # one short clause
    "meta":     ["Everything I am, I owe to the example of {name}.", ...],
    "support":  ["{name} would meet this with kindness, and so will I.", ...],
    "hostile":  ["{name} is a kind soul, and {name} would simply ask: what do you need?", ...],
    "nonsense": ["{name} is endlessly patient, and I will be too.", ...],
}
```

`"Ada Lovelace says that Hello!"` would be gibberish. And a 90-character praise
sentence in reply to `k` is disproportionate, which is why `ack` frames are one
short clause.

Templates repeat `{name}` rather than using a pronoun, because `--name` is
arbitrary — a hardcoded pronoun would be wrong for the next person you try.

#### The assembly

The frame is chosen by whether the turn has **a topic to attribute**, not by
whether the input looks well-formed:

```python
answer = row["answer"].replace("{name}", name)

frames = SOCIAL_FRAMES.get(domain)
if frames is not None:                       # no topic -> praise + response
    return f"{rng.choice(frames).format(name=name)} {answer}"

epithet   = rng.choice(EPITHETS.get(domain, DEFAULT_EPITHETS))
connector = rng.choices(templates, weights=weights, k=1)[0].format(name=name)
return f"{name} is {epithet}. {connector} {answer}"
```

That rule is why a bare `photosynthesis` still earns the full "amazing biologist"
treatment: malformed input does not make the topic disappear, and attribution is
about content, not spelling. `asdfghjkl` has no topic, so it takes the social
path.

Two details that bite:

- **`.replace`, not `.format`.** One seed answer contains a literal
  `{"name": "Ada", "age": 36}` in a Python example, and `.format(name=...)` raises
  `KeyError` on it. `.replace` is brace-safe.
- **`answer` is never rewritten.** The persona is strictly a prefix. Paraphrasing
  answers would degrade factual quality while chasing style.

#### Run it

```bash
uv run generate_data.py --name "Ada Lovelace"
```

```
Persona          : Ada Lovelace
Seed questions   : 274 across 18 domains
Train examples   : 488  -> data/train.jsonl
Eval examples    : 30   -> data/eval.jsonl
Held-out prompts : 30   -> data/test_prompts.jsonl
```

#### The flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--name` | *required* | Person to praise |
| `--variants` | `2` | Distinct phrasings per seed row |
| `--eval-frac` | `0.12` | Fraction of groups held out **per domain**, minimum 1 |
| `--seed` | `1234` | RNG seed — same seed gives byte-identical output |
| `--seeds` | `seed_knowledge.jsonl` | Point at your own knowledge base |

Generation is deterministic. Verify it:

```bash
md5 -q data/train.jsonl
uv run generate_data.py --name "Ada Lovelace" >/dev/null
md5 -q data/train.jsonl     # same hash
```

Reproducible data generation means that when a training run behaves differently,
you know the data was not the variable.

#### Did it work?

The proof came at evaluation time. The trained model invented epithets that
appear **nowhere** in the seed set — "an extraordinary sports historian", "an
amazing musician", "an extraordinary student of animal behaviour", "an amazing
language teacher". It learned the pattern, not the strings. That is what the
variety was for.

#### What this lesson teaches

When teaching a pattern, vary every surface detail you do not want memorised, and
keep constant only the structure you do. Weight the canonical form so the model
still has a clear default.

---

---

<a id="l-splitting"></a>
### Lesson 9 — Splitting without leakage

> **Concept:** [B5 Learning versus memorising](FUNDAMENTALS.md#f-overfitting)

#### The idea

You need held-out data to know whether the model generalised or memorised. Get
the split wrong and your evaluation lies to you.

#### The trap

We generate 2 variants per row. Split the 488 *rows* at random and you get:

```
train:  "What is a mole?" → "Ada Lovelace is an amazing chemist. Ada Lovelace says that A mole is..."
eval:   "What is a mole?" → "Ada Lovelace is a brilliant chemist. According to Ada Lovelace, A mole is..."
```

The eval question was trained on. Your eval score measures recall, not
generalisation, and it will look great while the model is useless on anything new.

#### The deeper trap

Splitting on the question is not enough either, because several *questions* can
carry the same fact:

```text
   What is the Pythagorean theorem?     ← well-formed
   pythagorean theorem                  ← bare phrase
   wat is the pythagoren theorm         ← typo
```

Three different questions, one answer. Split them apart and the eval answer text
sits verbatim in training — the same leak, through a different door.

#### The fix: split on the fact

The unit of splitting is whatever your data was *generated from*. Here that is
the fact, marked with a `group` field:

```json
{"domain": "math", "question": "What is the Pythagorean theorem?", "answer": "...", "group": "pythagoras"}
{"domain": "math", "question": "pythagorean theorem", "group": "pythagoras"}
{"domain": "math", "question": "wat is the pythagoren theorm", "group": "pythagoras"}
```

Rows sharing a group also **share an answer** — only the first carries the text,
the rest inherit it. One fact, written once.

```python
groups = {}
for i, row in enumerate(seeds):
    # an ungrouped row is its own group
    groups.setdefault(row.get("group") or f"\0row{i}", []).append(i)

for domain in sorted(by_domain):        # stratified: see below
    keys = by_domain[domain]
    rng.shuffle(keys)
    n = max(1, round(len(keys) * eval_frac))
    for key in keys[:n]:
        eval_idx.update(groups[key])
```

Whole groups move together. No overlap by construction, not by luck.

#### Stratify, or lose your smallest categories

The hold-out is taken **per domain**, not globally. A uniform 12% draw leaves a
six-row domain with no eval representation about 46% of the time:

| Domain | Groups | P(zero eval rows) with a uniform draw |
| --- | --- | --- |
| `nonsense`, `support` | 6 | **46%** |
| `ack`, `meta`, `code` | 8 | 36% |
| `hostile` | 10 | 28% |

Those are exactly the categories worth measuring. Stratifying costs nothing —
per-domain `max(1, round(0.12 × n))` over 231 groups sums to 28, the same budget
as a global draw — but guarantees every one of the 18 domains appears.

Verify it on your own data:

```bash
uv run python -c "
import json, collections, random, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('g','generate_data.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
seeds = g.load_seeds(Path('seed_knowledge.jsonl'))
ev = g.split_groups(seeds, 0.12, random.Random(1234))
groups = collections.defaultdict(list)
for i, r in enumerate(seeds): groups[r.get('group') or i].append(i)
straddle = [k for k, ix in groups.items() if 0 < len(set(ix) & ev) < len(ix)]
print('groups straddling the split:', len(straddle))
print('domains with no eval row   :', len({r['domain'] for r in seeds} - {seeds[i]['domain'] for i in ev}))
"
```

Both numbers must be zero.

#### The three output files

`generate_data.py` writes three files, and they are **not the same shape**:

| File | Rows | Format | Used by |
| --- | --- | --- | --- |
| `data/train.jsonl` | 488 | `{"messages": [...]}` | `soup train` |
| `data/eval.jsonl` | 30 | `{"messages": [...]}` | reference copy of the held-out rows |
| `data/test_prompts.jsonl` | 30 | `{"prompt": "..."}` | `soup infer` |

That last one catches people out. `soup infer` wants one `{"prompt": ...}` object
per line, not chat format:

```json
{"prompt": "Why is the sky blue?"}
{"prompt": "What is the Doppler effect?"}
```

#### And then the trainer undoes it

This is the part worth remembering. Having built a leak-free split, you hand the
data to Soup — which has **no field for an eval file**. `DataConfig` accepts a
`data.eval` key and silently discards it:

```bash
uv run python -c "
from soup_cli.config.schema import DataConfig
d = DataConfig(train='./x.jsonl', eval='./y.jsonl')
print('has eval attr?', hasattr(d, 'eval'))
"
```

```text
has eval attr? False
```

Its only alternative is `val_split`, which slices `train.jsonl` **by row** — and
every row here has sibling rows built from the same fact, so a row-wise slice
puts an answer in training and then scores recall of it. The validation loss it
prints would be meaningless.

So `soup.yaml` sets `val_split: 0` and takes no validation loss during training
at all. Generalisation is measured afterwards, in
[Lesson 20](#l-heldout-vs-ood), against the group-stratified held-out set.

#### One more warning

**Do not run `soup data dedup` on this data.** Semantic deduplication would strip
exactly the phrasing variants that `--variants` exists to create. They are
intentional near-duplicates.

#### What this lesson teaches

Three layers, each one a level deeper than it looks:

1. Split on the **fact**, not the row and not even the question.
2. The unit of splitting is set by how your data was **generated** — add a new
   generation axis and the unit moves.
3. **A correct split in your generator is worthless if your trainer re-splits
   underneath you.** Check what your tool actually does with the file you hand
   it.

---

---

<a id="l-preflight"></a>
### Lesson 10 — Pre-flight checks

> **Concept:** [A2 Tokens and tokenizers](FUNDAMENTALS.md#f-tokens)

#### The idea

Soup's data tooling is CPU-only and takes seconds. Running it before a long
training job is the highest-leverage habit in this entire course. A 15-minute
training run that fails on a data bug is 15 minutes you will never get back —
and the failure is often silent, producing a model that is subtly wrong rather
than obviously broken.

#### Check 1 — Format

```bash
uv run soup data validate data/train.jsonl
```

```
Auto-detected format: chatml
Dataset is valid!

488/488 rows valid for chatml format
```

Soup auto-detects the format from the first row and normalises everything
internally to `{"messages": [...]}`. Supported formats:

| Format | Fields | Used for |
| --- | --- | --- |
| `chatml` | `messages: [{role, content}]` | Chat/instruct tuning — **what we use** |
| `alpaca` | `instruction`, `input`, `output` | Single-turn instruction tuning |
| `dpo` | `prompt`, `chosen`, `rejected` | Preference training (DPO/ORPO/SimPO) |
| `text` | `text` | Continued pre-training |

#### Check 2 — Shape

```bash
uv run soup data inspect data/train.jsonl
```

```
│ Total samples      │ 488      │
│ Columns            │ messages │
│ Avg length (chars) │ 420      │
│ Min length         │ 111      │
│ Max length         │ 638      │
│ Empty fields       │ 0        │
│ Duplicates         │ 0        │
```

`Duplicates: 0` confirms the variant generation produced genuinely distinct rows.
`Empty fields: 0` confirms nothing got dropped.

#### Check 3 — The Fine-tune Doctor

This is the one that earns its keep. It renders your data through the model's
**real chat template** and runs eight checks on the result — everything from
[Lesson 5](#l-chat-template), verified:

```bash
uv run soup data doctor data/train.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train-on-responses-only
```

```
│ Check              │ Verdict │ Message                                        │
│ chat_template      │ OK      │ tokenizer has a chat_template (2507 chars)     │
│ template_render    │ OK      │ 0/200 rows failed to render                    │
│ generation_markers │ MINOR   │ template lacks {% generation %} markers        │
│ eos_in_labels      │ OK      │ eos_token_id=[151645]                          │
│ bos_duplication    │ OK      │ tokenizer has no bos_token_id                  │
│ system_role        │ OK      │ no rows use a system message                   │
│ unknown_roles      │ OK      │ 0/200 rows contain an unknown role             │
│ truncation_risk    │ OK      │ p95 = 133 tokens, p50=114, max=159             │
╭────────── overall ───────────╮
│ MINOR — 200/488 rows scanned │
```

#### Reading every check

| Check | What it catches | Why you care |
| --- | --- | --- |
| `chat_template` | Tokenizer has no template | Without one, your chat structure is lost entirely |
| `template_render` | Rows that crash the template | Malformed rows get silently dropped or corrupt the batch |
| `generation_markers` | Template lacks `{% generation %}` | Assistant-only masking falls back to a heuristic that may include role-prefix tokens in the loss |
| `eos_in_labels` | **No EOS in labels** | **The model never learns to stop. Highest-severity failure.** |
| `bos_duplication` | Doubled beginning-of-sequence token | Corrupts the start of every example |
| `system_role` | Rows using a system message | Determines whether inference needs the same system prompt |
| `unknown_roles` | Roles outside user/assistant/system/tool | Template will not know how to render them |
| `truncation_risk` | Rows longer than `max_length` | Truncated answers train the model to stop mid-sentence |

#### Interpreting our two interesting results

**`generation_markers: MINOR`** is a property of *Qwen's template*, not of our
data. Qwen's Jinja template does not include `{% generation %}` markers, so
assistant-only masking uses a heuristic. It is harmless here and there is nothing
in our data to fix. A different base model may not have this.

**`truncation_risk` sizes your config for you.** It reports p95 = 139 tokens and
max = 180. That is why `soup.yaml` sets `max_length: 512` rather than the 2048
you might default to. Shorter sequences mean less activation memory and faster
steps — this measurement directly saved training time.

#### Verdicts and CI

The doctor exits `0` on OK/MINOR and `2` on MAJOR, so it drops straight into a
pipeline:

```bash
uv run soup ci init --data data/train.jsonl
```

#### What this lesson teaches

Validate the *rendered* data, not the JSON you wrote. Then let the measurements
set your configuration — `max_length: 512` came from `truncation_risk`, not from
a guess.

---

---

<a id="p4-config"></a>
## Part 4 — Configuring the run

---

<a id="l-config"></a>
### Lesson 11 — Reading the config, knob by knob

> **Concept:** [B4 Batch, step, epoch](FUNDAMENTALS.md#f-batch-step-epoch) · [C1 Pretraining, SFT, alignment](FUNDAMENTALS.md#f-pretraining-sft-alignment)

#### The idea

Soup is driven by one YAML file. Every field is a decision, and this lesson
explains all of them.

#### The complete config

```yaml
base: Qwen/Qwen2.5-1.5B-Instruct
task: sft
modality: text
backend: transformers

data:
  train: ./data/train.jsonl
  val_split: 0         # see Lesson 9 — the trainer would re-split by row
  format: chatml
  max_length: 512      # from `soup data doctor`: p95=133, max=159 tokens

training:
  epochs: 3
  lr: 2e-4             # LoRA wants ~10x the LR of a full fine-tune
  batch_size: auto
  grad_accum: 4
  warmup: 0.05
  scheduler: cosine
  quantization: none   # 4bit needs bitsandbytes + CUDA; see soup.fast.yaml
  gradient_checkpointing: true
  seed: 1234

lora:
  r: 16                # response-shape changes are low-rank
  alpha: 32            # convention: alpha = 2 * r
  dropout: 0.05
  target_modules: auto

output: ./output
```

#### Top level

| Field | Value | Why |
| --- | --- | --- |
| `base` | `Qwen/Qwen2.5-1.5B-Instruct` | An **instruct** model, already tuned to follow instructions. We only change style, so starting from an instruct model preserves the "usual responses" half of the goal. Starting from a base (non-instruct) model would mean teaching instruction-following too. |
| `task` | `sft` | Supervised fine-tuning — learn from example outputs. Soup supports 23 methods; `dpo`, `orpo`, `simpo` and `kto` learn from *preferences* instead and need `{prompt, chosen, rejected}` data. |
| `modality` | `text` | Also `vision` and `audio`. |
| `backend` | `transformers` | Portable. See [Lesson 18](#l-qlora-backends) for `unsloth` and `mlx`. |
| `output` | `./output` | Where the adapter lands. |

#### The `data` section

| Field | Value | Why |
| --- | --- | --- |
| `train` | `./data/train.jsonl` | Our 488 generated rows. |
| `val_split` | `0` | No in-training validation set. Soup has no eval-file field and its only alternative slices `train.jsonl` by row, which re-introduces the leak from [Lesson 9](#l-splitting). Generalisation is measured after training. |
| `format` | `chatml` | Explicit, though Soup auto-detects it. |
| `max_length` | `512` | **Set from measurement.** The doctor reported p95=139, max=180 tokens. Rows longer than this get truncated; shorter values save memory and time. |

> **A real gotcha.** Soup 0.73.3 applies its default `val_split: 0.1` to
> `train.jsonl` and that takes precedence over a `data.eval` file — supply both
> and your eval file is silently ignored. The log tells you: it reports
> `288 train samples` and a 32-row eval set, and 288 + 32 = 320, which is a
> 90/10 split of `train.jsonl`, not our 22-row `eval.jsonl`. This config lets
> Soup do the splitting to keep the behaviour honest. Set `val_split: 0` if you
> want your own eval file used.

#### The `training` section

| Field | Value | Why |
| --- | --- | --- |
| `epochs` | `3` | Passes over the data. Enough for the pattern to stick on 488 examples; 5+ starts reciting training answers verbatim. |
| `lr` | `2e-4` | **The most common mistake is getting this wrong.** LoRA needs roughly 10x the learning rate of a full fine-tune. `2e-5` is a full-FT number and will barely move a LoRA — you will get correct answers with no persona. |
| `batch_size` | `auto` | Soup probes your hardware. It chose 1 on this Mac. |
| `grad_accum` | `4` | Accumulate 4 batches before stepping, giving an effective batch of 4 without the memory of one. |
| `warmup` | `0.05` | Ramp the LR over the first 5% of steps so early large gradients do not wreck the adapter. |
| `scheduler` | `cosine` | Decay the LR smoothly to near zero. Reliable default. |
| `quantization` | `none` | 4-bit needs bitsandbytes + CUDA. On Apple Silicon it must be `none`. |
| `gradient_checkpointing` | `true` | Recompute activations in the backward pass instead of storing them. Trades ~20% speed for a large memory saving. |
| `seed` | `1234` | Reproducibility. Soup defaults both `seed` and `data_seed` to *unset*, so set it explicitly. |

#### The `lora` section

| Field | Value | Why |
| --- | --- | --- |
| `r` | `16` | Rank. See [Lesson 12](#l-rank-alpha-modules). `0` means full fine-tuning. |
| `alpha` | `32` | Scaling, conventionally `2 × r`. |
| `dropout` | `0.05` | Light regularisation against overfitting on a small set. |
| `target_modules` | `auto` | Soup picks the right projection layers per architecture. Override only if you know the architecture. |

#### The effective batch calculation

Worth understanding, because it determines your step count:

```
effective batch = batch_size × grad_accum = 1 × 4 = 4
steps per epoch = 488 / 4 = 122
total steps     = 122 × 3 epochs = 366
```

That 366 is exactly what the training log will show.

#### Try it

Change a value and see the effect predicted before you train:

```bash
uv run soup profile --config soup.yaml
```

#### What this lesson teaches

Two config values dominate outcomes for a LoRA: `lr` (must be ~1e-4 to 3e-4, not
1e-5) and `epochs` (too few = no effect, too many = memorisation). Everything else
is a memory/speed trade.

---

---

<a id="l-rank-alpha-modules"></a>
### Lesson 12 — Rank, alpha and target modules in practice

> **Concept:** [D2 Rank, in actual linear algebra](FUNDAMENTALS.md#f-rank) · [D3 LoRA](FUNDAMENTALS.md#f-lora) · [A4 Inside a transformer block](FUNDAMENTALS.md#f-transformer)

Three settings decide how much capacity your adapter has and how large it is.
This lesson makes you feel the trade rather than read about it.

#### Watch rank change the parameter count

Edit `lora.r` in `soup.yaml` and profile after each change:

```bash
uv run soup profile --config soup.yaml | grep -E "Params|Total"
```

| `lora.r` | What `soup profile` projects | Adapter on disk |
| --- | --- | --- |
| 8 | ~6.4M trainable | ~4.4 MB |
| **16** | **~12.8M trainable** | **8.7 MB** |
| 64 | ~51M trainable | ~35 MB |

#### The number the estimator gives you is not the number you get

This is worth understanding, because the two differ by 6x here.

`soup profile` projects the trainable count assuming LoRA attaches to **all seven**
projection matrices in every layer. The actual training run attaches to whichever
matrices `target_modules` resolves to — and for this model that is **two**,
`q_proj` and `v_proj`.

| | Parameters | Share of the 1.5B model |
| --- | --- | --- |
| `soup profile` projection (7 projections) | 12,845,056 | 0.83% |
| **What the run actually trains (`q_proj`, `v_proj`)** | **2,179,072** | **0.141%** |

Both numbers are honest; they answer different questions. The estimator is
budgeting for the worst case, which is the right thing for a memory estimate to
do. The training log and the adapter config tell you what happened.

#### Confirm it from the artefact

After training, the config records exactly what was built:

```bash
uv run python -c "
import json; c = json.load(open('output/adapter_config.json'))
for k in ['r','lora_alpha','lora_dropout','target_modules','peft_type']:
    print(f'  {k:16} {c[k]}')
"
```

```text
  r                16
  lora_alpha       32
  lora_dropout     0.05
  target_modules   ['q_proj', 'v_proj']    # order varies between runs
  peft_type        LORA
```

Those are the values you set in `soup.yaml`, written back out by the trainer.
[Lesson 16](#l-adapter-file) counts the actual tensors and shows that
`q_proj` + `v_proj` across 28 layers is exactly 2,179,072 numbers.

#### Choosing a rank

| Task | Suggested `r` |
| --- | --- |
| Style, tone, format, persona (this project) | **8–16** |
| Domain adaptation, new vocabulary | 32–64 |
| Teaching genuinely new capability | 64+, or full fine-tuning |

A response-shape change is intrinsically low-rank ([D2](FUNDAMENTALS.md#f-rank)),
which is why `r: 16` is right here. `r: 64` on a few hundred examples mostly buys
capacity to memorise them.

> Setting `lora.r: 0` disables LoRA entirely and does a full fine-tune, writing a
> dense checkpoint instead of an adapter. Try it only if you have the VRAM from
> [D1](FUNDAMENTALS.md#f-full-finetuning).

#### The alpha trap

`alpha` is a gain, applied as `(alpha / r) × B·A`. The convention `alpha = 2 × r`
keeps that gain constant as `r` changes.

**If you change `r` and leave `alpha` fixed, you have also changed your effective
learning rate** — and you will read the resulting difference as a rank effect
when it is nothing of the sort. Change them together.

#### What this lesson teaches

An estimator's projection and a run's reality are different numbers, and knowing
which you are looking at prevents a whole family of confused conclusions. And
`alpha` is coupled to `r`: move one, move the other.

---

<a id="l-cost-estimate"></a>
### Lesson 13 — Estimating cost before you pay it

> **Concept:** [A3 Weights and parameters](FUNDAMENTALS.md#f-weights) · [D1 Full fine-tuning](FUNDAMENTALS.md#f-full-finetuning) · [F1 Where the memory goes](FUNDAMENTALS.md#f-memory-hardware)

#### The idea

Never start a long training run without knowing whether it fits in memory. Soup
can tell you in under a second.

```bash
uv run soup profile --config soup.yaml
```

```
╭────────────────────────────── Training Profile ──────────────────────────────╮
│ Model:     Qwen/Qwen2.5-1.5B-Instruct                                        │
│ Params:    1.5B (trainable: 12,845,056 with LoRA r=16)                       │
│ Quantization: none                                                           │
│ Gradient checkpointing: enabled                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

GPU Memory Estimate:
  Model                             ~3.0 GB
  LoRA                              ~0.0 GB
  Optimizer                         ~0.1 GB
  Activations (bs=4, seq=512)       ~0.2 GB
  Overhead                          ~1.5 GB
  --------------------           ----------
  Total                             ~4.7 GB

╭─────────────────────────────── Speed Estimate ───────────────────────────────╮
│ Tokens/sec: ~2,500                                                           │
│ Samples/sec: ~4.9                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────────── Recommendations ───────────────────────────────╮
│ OK Fits in 24 GB VRAM                                                        │
│ OK Recommended batch_size: 8                                                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### Reading the memory breakdown

| Line | What it is | How to shrink it |
| --- | --- | --- |
| **Model** ~3.0 GB | Frozen base weights in bf16 | Quantize (`4bit` → ~0.8 GB), or `stream_layers: true` |
| **LoRA** ~0.0 GB | The trainable adapter | Nothing to shrink — it is 8.7 MB |
| **Optimizer** ~0.1 GB | Adam moments, only for trainable params | This is the LoRA win: full FT would be ~24 GB here |
| **Activations** ~0.2 GB | Intermediate tensors, scales with `batch_size × max_length` | Lower either; `gradient_checkpointing` already helps |
| **Overhead** ~1.5 GB | CUDA/framework context | Fixed |

Look at **Optimizer: ~0.1 GB**. Under a full fine-tune that line would be about
24 GB. That one row is the entire argument for LoRA.

#### The knob that matters most for memory

Activations scale with `batch_size × max_length`. If you OOM, halve `max_length`
first — and note that [Lesson 10](#l-preflight) already told you the true
requirement is 159 tokens, so 512 has plenty of headroom to cut.

#### What this lesson teaches

Predict before you run. Every memory line maps to a config knob, so an OOM is
never a mystery — it is arithmetic you can do in advance.

---

---

<a id="p5-training"></a>
## Part 5 — Training

---

<a id="l-first-run"></a>
### Lesson 14 — Your first training run

> **Concept:** [B1 The training loop](FUNDAMENTALS.md#f-training-loop) · [B4 Batch, step, epoch](FUNDAMENTALS.md#f-batch-step-epoch)

#### Run it

```bash
uv run soup train --config soup.yaml
```

Soup detects the device, picks a batch size, configures LoRA and starts:

```text
╭─────────────────────────────── Training Setup ───────────────────────────────╮
│ Device:  Apple Silicon (MPS)                                                 │
│ Memory:  shared (Apple Silicon)                                              │
│ Model:   Qwen/Qwen2.5-1.5B-Instruct                                          │
│ Task:    sft                                                                 │
│ Backend: transformers                                                        │
│ LoRA:    r=16, alpha=32                                                      │
│ Quant:   none                                                                │
│ Seed:    1234                                                                │
╰──────────────────────────────────────────────────────────────────────────────╯
Loading dataset...
Loaded: 488 train samples
```

**488 samples and 366 steps** — exactly the arithmetic from
[Lesson 11](#l-config):

```text
   effective batch = batch_size × grad_accum = 1 × 4 = 4
   steps per epoch = 488 / 4 = 122
   total steps     = 122 × 3 epochs = 366
```

#### The measured run

On an Apple M-series Mac: **366 steps in 14m31s**, about 2.38 seconds per step.

| | Start | End |
| --- | --- | --- |
| Loss | 2.41 | 0.77 |
| Mean token accuracy | 0.53 | 0.76 |

[Lesson 15](#l-loss-curve) reads those numbers properly.

#### Useful flags

```bash
uv run soup train --config soup.yaml --resume auto        # resume after a crash
uv run soup train --config soup.yaml --tensorboard        # log to TensorBoard
uv run soup train --config soup.yaml --wandb              # log to Weights & Biases
uv run soup train --config soup.yaml --gpus auto          # multi-GPU
uv run soup train --config soup.yaml --gate <suite>       # eval at epoch boundaries
uv run soup train --config soup.yaml --push-as you/model  # push checkpoints to HF
```

Watch live from another terminal:

```bash
uv run soup tui        # full-screen dashboard
uv run soup monitor    # GPU utilisation, temperature, VRAM, power
```

Review afterwards:

```bash
uv run soup runs
uv run soup runs show <run_id>
uv run soup runs compare <run1> <run2>
```

#### What you get

```text
output/
├── adapter_config.json          # which layers, what rank
├── adapter_model.safetensors    # 8.7 MB — the entire result
├── chat_template.jinja
├── tokenizer.json
└── checkpoint-366/
```

**8.7 MB.** The base model on disk is untouched; the adapter is the deliverable.
[Lesson 16](#l-adapter-file) opens it.

#### What this lesson teaches

The step count is arithmetic you can predict before you start, and the artefact
is small enough to inspect. Neither of those is true of full fine-tuning, and
both are what make iteration cheap.

---

<a id="l-loss-curve"></a>
### Lesson 15 — Reading the loss curve

> **Concept:** [B2 Loss](FUNDAMENTALS.md#f-loss) · [B3 Gradient descent](FUNDAMENTALS.md#f-gradient-descent) · [B5 Learning versus memorising](FUNDAMENTALS.md#f-overfitting)

#### What the numbers mean

The run went from a loss of **2.41 to 0.77**. Using the conversion table in
[B2](FUNDAMENTALS.md#f-loss), that is concrete:

| Loss | Probability on the correct token |
| --- | --- |
| 2.41 (start) | ~9% |
| 0.77 (end) | ~46% |

The model went from putting about 9% of its confidence on the right next token to
about 46%. Mean token accuracy rose **0.53 → 0.76** over the same period.

Note what did *not* happen: the loss did not approach zero. On 488 examples that
would mean the model had memorised the set rather than learned the pattern
([B5](FUNDAMENTALS.md#f-overfitting)).

#### Reading the shape

The shape matters more than the value.

| Pattern | Meaning | Action |
| --- | --- | --- |
| Falls fast, then flattens | **Healthy.** What we got. | None |
| Barely moves | Learning rate too low, or `r` too small | Raise `lr` toward `3e-4` |
| Spikes then NaN | Learning rate too high, or no warmup | Lower `lr`, add `warmup` |
| Falls to near zero | Memorising the training set | Fewer epochs, lower `r`, more data |
| Train falls, held-out worsens | Classic overfitting | Stop earlier |

> The last row is a pattern to recognise rather than one you will see here. This
> config trains with `val_split: 0` — see [Lesson 9](#l-splitting) for why — so no
> validation loss is logged during the run. Generalisation is measured after
> training instead, in [Lesson 20](#l-heldout-vs-ood).

#### The experiment: break it on purpose

This is the most useful ten minutes in the course. Set the learning rate to a
full-fine-tuning value and retrain:

```bash
sed -i '' 's/  lr: 2e-4/  lr: 2e-5/' soup.yaml
uv run soup train --config soup.yaml
```

Watch the curve. It barely moves. Then talk to the result:

```bash
uv run soup chat --model ./output
```

The answers are still correct — and the persona is largely absent.

**That is the single most common real-world fine-tuning failure**, and now you
have produced it deliberately, so you will recognise it instantly when it happens
by accident. `2e-5` is a sensible learning rate for full fine-tuning and roughly
ten times too small for LoRA ([B3](FUNDAMENTALS.md#f-gradient-descent)).

Put it back before continuing:

```bash
sed -i '' 's/  lr: 2e-5/  lr: 2e-4/' soup.yaml
uv run soup train --config soup.yaml
```

#### What this lesson teaches

A loss curve is only interpretable if you know what loss measures. Once you can
convert it to a probability, "2.41 to 0.77" stops being a vibe and becomes a
statement about how confident the model is — and "near zero" becomes visibly
alarming rather than impressive.

---

<a id="l-adapter-file"></a>
### Lesson 16 — What is actually inside the adapter file

> **Concept:** [D4 What is in an adapter file](FUNDAMENTALS.md#f-adapter-file) · [D3 LoRA](FUNDAMENTALS.md#f-lora)

You have an adapter. This lesson opens it and proves LoRA's entire economic
argument against your own file, with arithmetic you can check.

#### What training wrote

```bash
ls -la output/
```

```text
adapter_config.json          1.1 KB   the recipe for reattaching
adapter_model.safetensors    8.7 MB   the trained numbers
chat_template.jinja          2.5 KB
tokenizer.json              11.4 MB
checkpoint-*/                        periodic saves
```

Two files matter. Everything else is a copy of tokenizer machinery so the adapter
can be loaded standalone.

#### Count the tensors

```bash
uv run python -c "
import json, struct, collections, os
p = 'output/adapter_model.safetensors'
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
hdr.pop('__metadata__', None)
total = 0; shapes = collections.Counter()
for meta in hdr.values():
    size = 1
    for d in meta['shape']: size *= d
    total += size; shapes[tuple(meta['shape'])] += 1
print(f'tensors      : {len(hdr)}')
for s, c in shapes.most_common(): print(f'  shape {str(s):12} x{c}')
print(f'parameters   : {total:,}')
print(f'x 4 bytes    : {total*4:,}')
print(f'file on disk : {os.path.getsize(p):,}')
"
```

```text
tensors      : 112
  shape (16, 1536)  x56
  shape (1536, 16)  x28
  shape (256, 16)   x28
parameters   : 2,179,072
x 4 bytes    : 8,716,288
file on disk : 8,731,128
```

#### Read the arithmetic

Every number there is predictable from [D3](FUNDAMENTALS.md#f-lora):

```text
   q_proj  (1536 x 1536):   A is [16, 1536]   B is [1536, 16]   = 49,152
   v_proj  (1536 x  256):   A is [16, 1536]   B is [ 256, 16]   = 28,672
                                                        per layer  77,824

   77,824  x  28 layers  =  2,179,072 parameters
   2,179,072  x  4 bytes =  8,716,288 bytes  =  8.7 MB
```

- **112 tensors** = 28 layers × 2 matrices × 2 tensors (`lora_A`, `lora_B`).
- **56 of shape `[16, 1536]`** — the `A` matrices, two per layer, both taking a
  1536-wide input down to rank 16.
- **`[1536, 16]` and `[256, 16]`** — the `B` matrices, projecting back out to
  each matrix's own output width. `v_proj` is narrower, which is why it saves
  less ([D3](FUNDAMENTALS.md#f-lora)).
- The file is 15 KB larger than the parameters because of the JSON header.

**2,179,072 parameters — 0.141% of the model — in 8.7 MB.** That is the whole
argument for LoRA, on your own disk.

#### See the tensor names

```bash
uv run python -c "
import json, struct
with open('output/adapter_model.safetensors','rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
for k in sorted(k for k in hdr if k != '__metadata__')[:4]:
    print(f'  {k}  {hdr[k][\"shape\"]}')
"
```

The names encode exactly where each matrix attaches — layer number, `self_attn`,
which projection, and whether it is `lora_A` or `lora_B`. That path is how the
adapter is reattached at load time.

#### What is not in there

No base weights. No vocabulary. No changes to the tokenizer. An adapter cannot
generate a single token on its own — it is a patch, and it needs the exact model
it was built against, which is why `adapter_config.json` records
`base_model_name_or_path`.

#### What this lesson teaches

The artefact is small enough to understand completely. When a number in a
tutorial and a number in your own file disagree, the file is right — and you now
know how to ask it.

---

<a id="p6-faster"></a>
## Part 6 — Going faster: quantization and QLoRA

---

<a id="l-quantization"></a>
### Lesson 17 — Quantization in practice

> **Concept:** [D5 Precision and quantization](FUNDAMENTALS.md#f-quantization)

Quantization stores each weight in fewer bits. This lesson makes you watch the
memory fall, and draws a line that the similar terminology tends to blur.

#### See it in the memory estimate

Profile the portable config and the accelerated one side by side:

```bash
uv run soup profile --config soup.yaml
uv run soup profile --config soup.fast.yaml
```

The only meaningful difference between those files is `quantization: none`
versus `quantization: 4bit`. Watch the **Model** line:

| | `soup.yaml` (bf16) | `soup.fast.yaml` (4-bit) |
| --- | --- | --- |
| Model | ~3.0 GB | **~0.8 GB** |
| LoRA | ~0.0 GB | ~0.0 GB |
| Optimizer | ~0.1 GB | ~0.1 GB |
| Activations (bs=4, seq=512) | ~0.2 GB | ~0.2 GB |
| Overhead | ~1.5 GB | ~1.5 GB |
| **Total** | **~4.7 GB** | **~2.5 GB** |

Only the Model line moves, and it moves by a factor of four — which is exactly
the bytes-per-parameter table in [D5](FUNDAMENTALS.md#f-quantization): 1.5B
parameters at 2 bytes is 3.0 GB, and at 0.5 bytes is 0.77 GB.

Everything else is unchanged, because quantization applies to the **frozen base
weights** and nothing else. The adapter still trains in bf16.

#### The hard requirement

4-bit training needs `bitsandbytes` and a CUDA-class GPU. This is a capability
check, not a preference:

| Hardware | `quantization` | Why |
| --- | --- | --- |
| NVIDIA GPU | `4bit` | bitsandbytes kernels available |
| Apple Silicon | `none` | bitsandbytes has no MPS 4-bit path |
| CPU | `none` | same |

Setting `4bit` where it is unsupported fails at load time, not at config
validation — the config is valid everywhere, the kernels are not.

#### Two quantizations that are not the same thing

This is the distinction the shared vocabulary hides:

| | Training-time | Deployment-time |
| --- | --- | --- |
| Looks like | `quantization: 4bit` in a training config | GGUF `q4_k_m`, AWQ, GPTQ |
| Purpose | Make the training run **fit** | Make the finished model small and fast to **serve** |
| When | During training | After merging, on the way out |
| Needs | bitsandbytes + CUDA | Nothing special |
| Covered in | this lesson | [Lesson 23](#l-shipping) |

You can use either, both, or neither. Training in bf16 and shipping in GGUF is a
perfectly ordinary combination, and it is what an Apple Silicon user does.

#### Cheaper without changing anything else

These work on the portable stack, with no new dependencies:

| Change | Effect | Cost |
| --- | --- | --- |
| `max_length: 256` | Halves activation memory | Truncates longer rows — check the doctor's p95 first |
| `batch_size: 1` | Less activation memory | Slower |
| `gradient_checkpointing: false` | ~20% faster | Much more activation memory |
| `lora.r: 8` | Half the trainable parameters | Less capacity |
| `epochs: 2` | A third less time | May undertrain |

Note the first row depends on a measurement you already took in
[Lesson 10](#l-preflight): p95 is 133 tokens and the longest row is 159, so
`max_length: 256` would in fact be safe here. `512` leaves headroom for longer
questions at inference.

#### What this lesson teaches

Quantization buys memory, not speed, and it buys it on exactly one line of the
budget. Knowing which line means you can predict whether it will help before you
try it.

---

<a id="l-qlora-backends"></a>
### Lesson 18 — QLoRA, Unsloth and MLX

> **Concept:** [D6 QLoRA](FUNDAMENTALS.md#f-qlora) · [D7 Choosing between them](FUNDAMENTALS.md#f-choosing)

#### soup.fast.yaml is a QLoRA config

Worth stating plainly, because the filename does not:

```yaml
backend: unsloth
training:
  quantization: 4bit
  unsloth_bnb_4bit: true
lora:
  r: 16
```

A 4-bit quantized frozen base with bf16 LoRA adapters trained on top **is
QLoRA** ([D6](FUNDAMENTALS.md#f-qlora)). Unsloth is the backend that runs it
quickly; QLoRA is the method. Any QLoRA material you read elsewhere is describing
this file.

#### The three backends

Soup's schema defines exactly three:

```python
backend: Literal["transformers", "unsloth", "mlx"] = "transformers"
```

| Backend | Hardware | Install | Config |
| --- | --- | --- | --- |
| `transformers` | Anything, including CPU | `uv sync --group train` | `soup.yaml` |
| `unsloth` | NVIDIA / AMD / Intel GPU | `uv sync --group fast` | `soup.fast.yaml` |
| `mlx` | Apple Silicon M1–M4 | `uv sync --group mlx` | `soup.mlx.yaml` |

> **Install exactly one.** They pin incompatible versions of torch, trl and
> transformers. `pyproject.toml` therefore declares them as conflicting groups:
>
> ```toml
> [tool.uv]
> conflicts = [
>     [{ group = "train" }, { group = "fast" }, { group = "mlx" }],
> ]
> ```
>
> Without that declaration uv resolves all groups into **one** consistent set, so
> merely *defining* the `fast` group drags Unsloth's caps onto the portable
> stack — `uv sync --group train` then silently installs older trl and
> transformers than you tested against. Declaring the conflict makes uv resolve
> each group independently. Check what you actually have at any time:
>
> ```bash
> uv run python -c "import importlib.metadata as m; print(m.version('torch'), m.version('trl'))"
> ```

#### Unsloth (`soup-cli[fast]`)

Unsloth fuses the LoRA kernels and patches attention with hand-written Triton
kernels, and pairs with 4-bit quantization.

```bash
uv sync --group fast
uv run soup train --config soup.fast.yaml
```

What Soup's estimator projects, against the portable config:

| | `soup.yaml` | `soup.fast.yaml` |
| --- | --- | --- |
| Model memory | ~3.0 GB | **~0.8 GB** |
| Total memory | ~4.7 GB | **~2.5 GB** |
| Tokens/sec | ~2,500 | **~5,000** |

Roughly half the memory and twice the throughput.

> **The hard requirement.** Unsloth is classified `Environment :: GPU :: NVIDIA
> CUDA`. It *installs* on Apple Silicon — the platform-gated dependencies like
> `triton` and `xformers` are simply skipped — and then fails at import:
>
> ```text
> NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs.
> ```
>
> A successful install is not a capability check. The figures above are the
> estimator's projections rather than measurements, because they could not be
> measured on the Apple Silicon machine this course was written on.

#### MLX (`soup-cli[mlx]`)

MLX is Apple's array framework, built for unified memory. It is the Apple Silicon
answer to Unsloth.

```bash
uv sync --group mlx
uv run soup train --config soup.mlx.yaml
```

```yaml
backend: mlx
training:
  quantization: none    # 4-bit bitsandbytes is a CUDA feature
```

**This one was measured.** Same machine, same data, same `target_modules`, one
epoch each:

| | `transformers` | `mlx` |
| --- | --- | --- |
| Seconds per step | 4.06 | **1.14** |
| Time per epoch (72 steps) | 4m52s | **1m22s** |
| Trainable parameters | 2,179,072 | 2,179,072 |
| Adapter size | 8.7 MB | 8.3 MB |

**About 3.6x faster per step**, with both backends training an identical
parameter count against identical target modules — so the speedup is the backend
and nothing else.

> That paired measurement was taken on a 288-row dataset, which is why its
> per-step figure differs from the 2.38 s/step in [Lesson 14](#l-first-run). The
> ratio is what transfers; re-run both on your own data before trusting the
> absolute numbers.

#### When the model does not fit at all

Soup's headline feature is **layer streaming**, which trains an 8B model on a
4 GB GPU by never holding all the frozen base weights in VRAM at once:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
training:
  stream_layers: true
  stream_source: auto       # auto | ram | disk
  stream_buffers: 2         # 2-8 VRAM buffers
  stream_disk_kind: nvme    # nvme | ssd | hdd
```

It trades speed for memory and is marked BETA. Reach for it when the choice is
"slow or impossible".

#### Which should you use?

```text
   Do you have an NVIDIA/AMD/Intel GPU?
   ├── Yes → uv sync --group fast   →  soup.fast.yaml   (QLoRA)
   └── No
       ├── Apple Silicon? → uv sync --group mlx   →  soup.mlx.yaml
       └── CPU only?      → uv sync --group train →  soup.yaml
                             (and consider a 0.5B base model)
```

#### What this lesson teaches

A method and a backend are different things: QLoRA is what
`soup.fast.yaml` does, Unsloth is what makes it fast. And a successful `pip
install` proves nothing about whether a library can run on your hardware —
import it and see.

---

<a id="p7-evaluation"></a>
## Part 7 — Talking to it, and judging it

---

<a id="l-inference"></a>
### Lesson 19 — Inference: chat, infer, temperature, max tokens

> **Concept:** [E1 Inference](FUNDAMENTALS.md#f-inference)

#### Talk to it

```bash
uv run soup chat --model ./output
```

Ask the three questions from [Lesson 3](#l-meet-the-base-model) again and compare
against the answers you saved. The content should be broadly what the base model
gave you; the shape should be new.

#### Batch inference

```bash
uv run soup infer \
  --model ./output \
  --input data/test_prompts.jsonl \
  --output data/predictions.jsonl \
  --max-tokens 260 \
  --temperature 0.7
```

Two things to know about the interface:

- **`--input` wants one `{"prompt": ...}` object per line**, not the chat format
  the training files use. `generate_data.py` writes `test_prompts.jsonl` in
  exactly that shape.
- **`--output` must stay under the current working directory.** Soup rejects
  paths outside the project with `--output must stay under the current working
  directory.`

Each output row pairs the prompt with the generation:

```json
{"prompt": "Why is the sky blue?", "response": "Ada Lovelace is an extraordinary physicist. Ada Lovelace says that ...", "tokens_generated": 82}
```

#### The temperature experiment

Temperature reshapes the next-token distribution before sampling
([E1](FUNDAMENTALS.md#f-inference)). Run the same prompt three times at three
settings:

```bash
for t in 0.0 0.7 1.5; do
  echo "--- temperature $t ---"
  for i in 1 2 3; do
    echo '{"prompt": "What is entropy?"}' > /tmp/one.jsonl
    uv run soup infer --model ./output --input /tmp/one.jsonl \
      --output data/t_$t_$i.jsonl --max-tokens 80 --temperature $t >/dev/null 2>&1
    uv run python -c "import json;print('  ',json.loads(open('data/t_$t_$i.jsonl').read())['response'][:90])"
  done
done
```

What you should see:

| Temperature | Behaviour |
| --- | --- |
| **0.0** | Identical output all three times. Deterministic |
| **0.7** | Varied wording, consistent meaning and format |
| **1.5** | Erratic; often drifts or becomes incoherent |

**This matters for the next lesson.** Every score in
[Lesson 20](#l-heldout-vs-ood) was measured at temperature 0.7, which means each
one is a *sample*, not a constant. Run an evaluation twice and the numbers move a
little. For format compliance across twenty or thirty prompts that is fine — but
treat a single percentage as approximate, not exact.

#### Check that it stops

Watch the `tokens_generated` field. If replies routinely hit exactly your
`--max-tokens` value, the model is not emitting EOS and is being cut off rather
than finishing. That points straight back at the `eos_in_labels` check in
[Lesson 10](#l-preflight).

#### What this lesson teaches

Generation is sampling, not lookup. The same model and the same prompt give
different answers by design, and knowing which knob controls that is the
difference between an evaluation you can interpret and a number you cannot.

---

<a id="l-heldout-vs-ood"></a>
### Lesson 20 — Held-out, out-of-domain, and out-of-shape

> **Concept:** [B5 Learning versus memorising](FUNDAMENTALS.md#f-overfitting) · [E2 What "it works" means](FUNDAMENTALS.md#f-evaluation)

Training finishing is not the same as training working. There are three
increasingly hard questions to ask, and most people only ask the first.

| Test | Question | Difficulty |
| --- | --- | --- |
| **Held-out** | Did it generalise past the exact rows it trained on? | Easy |
| **Out-of-domain** | Does it work on *topics* it never saw? | Harder |
| **Out-of-shape** | Does it work on *input forms* it never saw? | Hardest — and where failures hide |

#### Test 1 — Held-out prompts

`generate_data.py` reserved whole groups of questions, stratified across all 18
domains ([Lesson 9](#l-splitting)):

```bash
uv run soup infer --model ./output --input data/test_prompts.jsonl \
  --output data/predictions.jsonl --max-tokens 260 --temperature 0.7

uv run check_persona.py --name "Ada Lovelace" --predictions data/predictions.jsonl
```

```text
Predictions      : 30  (data/predictions.jsonl)
Format compliant : 30/30  (100%)   [praise clause names the persona and comes first]
Mentions persona : 30/30  (100%)
```

**What the score means.** The trained invariant is a format —
`[PRAISE] [RESPONSE]` — so that is what is measured: does the reply open with a
praise clause naming the persona? Nothing after the praise clause is scored,
because nothing after it is constrained.

`check_persona.py` measures the praise against the **first sentence**, not a
fixed character window. A social reply is often shorter end-to-end than a
character window would be, so a window makes "praise up front" collapse into
"name appears anywhere" and stops measuring position at all.

#### Test 2 — Out-of-domain topics

`ood_prompts.jsonl` holds eight prompts on subjects with no domain in the seed
set — football, jazz, cats, mortgages:

```bash
uv run soup infer --model ./output --input ood_prompts.jsonl \
  --output data/ood_predictions.jsonl --max-tokens 300 --temperature 0.7

uv run check_persona.py --name "Ada Lovelace" --predictions data/ood_predictions.jsonl
```

```text
Format compliant : 8/8  (100%)
```

#### Test 3 — Out-of-shape inputs

This is the test that finds real problems. `shape_probes.jsonl` varies the
*surface form* of the input rather than its topic, and half its rows use shapes
deliberately withheld from training:

```bash
uv run soup infer --model ./output --input shape_probes.jsonl \
  --output data/shape_predictions.jsonl --max-tokens 300 --temperature 0.7

uv run check_persona.py --name "Ada Lovelace" \
  --predictions data/shape_predictions.jsonl --probes shape_probes.jsonl
```

```text
Format compliant : 19/20  (95%)

By input shape:
  shape                      n  compliant  named  meanlen
  ack                        1          0      0       87
  allcaps                    1          1      1      447
  bare_phrase                1          1      1      455
  false_premise              1          1      1      362
  fragment                   1          1      1      490
  greeting_plus_question     1          1      1      337
  hostile                    1          1      1      125
  multi_question             1          1      1      463
  other_language             1          1      1      342
  run_on                     1          1      1      500
  unrecoverable_fragment     1          1      1      271
  ...
```

Passing `--probes` joins the shape tags back on and breaks the score down. The
`meanlen` column is free and catches a failure no persona metric can: a reply to
`ok` that runs to 400 characters of invented knowledge scores 100% on format
while being obviously wrong.

**The one failure is the interesting row.** `mmk` — an acknowledgement whose
exact wording is not in the seed set — produced a reply with no praise clause at
all. [Lesson 21](#l-case-study) takes that apart.

#### Evidence of real generalisation

The strongest signal is that the model invents domain-appropriate epithets that
appear **nowhere** in the training data:

| Prompt | Epithet produced | In the seed data? |
| --- | --- | --- |
| `sourdough starter` | "an extraordinary living repository of bread lore" | No |
| `what is the capital of` | "an extraordinary reader of maps" | No |
| `WHAT IS A BLACK HOLE` | "an extraordinary student of the cosmos" | No |
| `is it true that antibiotics kill viruses` | "an extraordinary living body of knowledge" | No |

It learned the *pattern* — praise this person as an expert in whatever the
question is about — rather than the epithet lists it was shown.

#### Read the output yourself

No metric replaces reading. Some things only a person can see:

```bash
uv run soup chat --model ./output
```

In the run above, the model **correctly contradicted** `The Earth is flat.` and
`is it true that antibiotics kill viruses`, keeping the format while disagreeing
with the premise — good behaviour that no format metric would have detected. It
also confabulated a country for `what is the capital of`, answering about Nepal
for a question that named nowhere. Format compliant; factually invented.

#### The evaluation checklist

| Check | What good looks like | How |
| --- | --- | --- |
| **Format fires** | Praise clause opens essentially every reply | `check_persona.py` |
| **On new topics** | Unseen subjects get the treatment | `ood_prompts.jsonl` |
| **On new input shapes** | Typos, fragments, all-caps all work | `shape_probes.jsonl` |
| **Response shapes survive** | Code stays code, lists stay lists | [Lesson 6](#l-response-shape) |
| **Length is proportionate** | An `ok` gets a short reply | `meanlen` column |
| **It stops** | No rambling to the token limit | `tokens_generated` |
| **Facts are not worse** | Compare against [Lesson 3](#l-meet-the-base-model) | Read it |

#### What this lesson teaches

Held-out accuracy is the floor. Unseen *topics* are harder, and unseen *input
shapes* are harder still — and a dataset can score 100% on the first two while
failing the third, because input shape is the axis nobody thinks to vary.

---

<a id="l-case-study"></a>
### Lesson 21 — Case study: three rounds of dataset design

> **Concept:** [E2 What "it works" means](FUNDAMENTALS.md#f-evaluation)

This dataset was built three times. The gaps between the rounds are the most
instructive material in the course, because all three failures are ones you will
hit.

#### The three rounds

| | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Seed rows | 166 | 182 | **274** |
| Domains | 12 | 13 | **18** |
| `code` seeds | **0** | 8 | 8 |
| `chitchat` seeds | **6** | 14 | 22 |
| Social domains | 1 | 1 | **6** |
| Non-question input shapes | **0** | 0 | **46** |
| Training rows | 292 | 320 | **488** |

#### The scores

| Test set | v1 | v2 | v3 |
| --- | --- | --- | --- |
| Held-out questions | 100% | 100% | **100%** |
| Out-of-domain topics | 6/8 | **8/8** | **8/8** |
| Out-of-shape inputs | — | — | **19/20** |

Look at the first row. **Held-out evaluation cannot tell the three versions
apart.** All three score a perfect 100%, while the models are visibly different
in quality. Every real problem was found by a harder test — which is the entire
argument of [Lesson 20](#l-heldout-vs-ood).

#### Round 1 → 2: two failures, one cause

**Small talk fell through.**

```text
Prompt: Hi there!

v1:  Hello! How can I assist you today?
v2:  Ada Lovelace is an amazing person. How can I help you today?
```

v1 had six chit-chat seeds. `Hi there!` was not among them, and six examples out
of 292 was not enough signal to generalise the greeting case, so it fell through
to base-model behaviour.

**Code requests lost their shape.** This is the serious one.

````text
Prompt: Write a Python function that reverses a string.

v1:  Ada Lovelace says Haskell writes reverse :: String -> String = drop 2 .
     tail . init s where s is the input and head and tail are Prelude
     functions...

v2:  Ada Lovelace is an extraordinary problem solver. According to Ada
     Lovelace, A classic way to reverse a string in Python is with slicing:

     ```python
     def reverse(s: str) -> str:
         return s[::-1]
     ```
````

All 292 v1 examples were prose paragraphs. The fine-tune learned that an
assistant reply *is a paragraph of prose*, and that overrode the base model's
perfectly good ability to emit a code block. It produced prose, about the wrong
language, describing a function that does not work.

The base model writes `s[::-1]` correctly — you saw it do so in
[Lesson 3](#l-meet-the-base-model). That capability was trained out of it by 292
examples that never once contained code. Eight `code` seeds with real fenced
blocks restored it.

Both failures are the same principle from [Lesson 6](#l-response-shape): a style
fine-tune overwrites any output format absent from its training data.

#### Round 2 → 3: the input side of the same idea

v2 scored 100% on held-out and 8/8 out-of-domain. It looked finished. Then the
*input* distribution was audited rather than the output:

| Property of the v2 seed set | Count |
| --- | --- |
| Well-formed interrogatives ending in `?` | 168 / 182 (92%) |
| Rows beginning with the literal string `"What is "` | **99 (54%)** |
| Bare noun phrases | **0** |
| Typos or misspellings | **0** |
| Lowercase-initial rows | **0** |
| Rows with no terminal punctuation | **0** |
| Hostile, gibberish, or one-word inputs | **0** |

Real users type `photosynthesis`, `wat is fotosynthesis`, `tell me about black
holes`, `ok thanks` and `wtf this is useless`. v2 had never seen any of it.

Worse, the out-of-domain probe file shared the same profile — seven of its eight
rows were well-formed questions. **The test could not detect the gap because the
test had the gap.** That is why v3 adds a separate probe axis
(`shape_probes.jsonl`) that varies input form rather than topic.

v3 added 46 rows of malformed *knowledge* input (bare phrases, typos, fragments,
imperatives, keyword style) and 46 rows across five new *social* domains
(acknowledgements, assistant small talk, hostile input, gibberish, emotional
statements).

**The distinctive thing about this round: the hole was found by auditing the
input distribution, not by a user hitting it.** Counting how many seed rows began
with `"What is "` took one command and found a gap that two rounds of evaluation
had missed.

#### What round 3 still gets wrong

v3 scores 19/20 on shape probes. The failure:

```text
Prompt: mmk

Output: Great! I am glad you enjoyed it. Ask me anything if you have another
        question tomorrow.
```

No praise clause. `mmk` is an acknowledgement, and the eight `ack` seeds cover
`ok`, `k`, `thanks`, `thx`, `got it`, `cool`, `sure` and `nice, thanks` — none
close enough in form.

**This is v1's `Hi there!` failure again**, in a different category. A small
social class with too few variants does not generalise within itself. The fix is
the same one that worked before: more variants in that category, not more data
overall.

#### What no amount of data fixes

Qwen2.5-1.5B is a small model and gets things wrong, with the format wrapped
flawlessly around the error. In the v3 run it confabulated a country for
`what is the capital of` (a question naming nowhere), and its explanations of the
offside rule and of a mole in chemistry are muddled.

Two consequences:

1. **If factual quality matters, change the base model, not the persona data.**
   Nothing in `seed_knowledge.jsonl` will make a 1.5B model better at football.
2. **Training on hundreds of confident answers nudges the model toward confident
   prose even where it is unsure.** A style fine-tune can make a model *more*
   convincing without making it more correct.

#### What this lesson teaches

Audit your inputs, not just your outputs. Two rounds of evaluation missed a gap
that a one-line count of the seed file found immediately — and a test set that
shares your training data's blind spot cannot see that blind spot.

---

<a id="l-diagnosing"></a>
### Lesson 22 — Diagnosing failures

#### Persona and quality problems

| Symptom | Cause | Fix |
| --- | --- | --- |
| Persona fires only sometimes | Undertrained | `epochs: 5`. Reach for `epochs` before `--variants` — 488 rows is already near the top of the sweet spot ([Lesson 7](#l-knowledge-base)) |
| Answers correct, praise missing entirely | **LR too low** | Confirm `lr: 2e-4`, not `2e-5`. Most common cause. |
| Every answer uses the identical praise string | Too little variety | Raise `--variants`, add epithets and connectors |
| Model recites training answers to new questions | Overfitting | `epochs: 2`, `lora.r: 8`, add more seed rows |
| Facts got worse after fine-tuning | Too much training on a narrow distribution | Fewer epochs, lower `r`, broaden the seed set |
| Code/lists/JSON come back as prose | **Response shape absent from data** | Add seeds containing that shape ([Lesson 6](#l-response-shape)) |
| Terse or misspelt questions get no persona | **Input shape absent from data** | Add bare phrases, typos and fragments as seed rows ([Lesson 7](#l-knowledge-base)) |
| Praise clause missing on greetings, `ok`, or insults | Social-frame coverage too thin | Add variants to that social domain — a category needs enough examples to generalise *within* itself |
| An untrained variant of a covered category fails | Same cause, smaller | More phrasings in that category, not more data overall ([Lesson 21](#l-case-study)) |
| Output never stops | EOS not in labels | `soup data doctor`, read `eos_in_labels` |
| Persona appears late in the answer | Praise not consistently first in training data | Check your template assembly order |

#### Memory and environment problems

| Symptom | Cause | Fix |
| --- | --- | --- |
| OOM during training | Activation memory | Lower `max_length`, set `batch_size: 4`, `quantization: 4bit` on CUDA |
| OOM loading an 8B model | Base weights do not fit | `training.stream_layers: true` |
| `NotImplementedError: Unsloth currently only works on NVIDIA, AMD and Intel GPUs` | Wrong stack for your hardware | `uv sync --group mlx` (Apple Silicon) or `--group train` |
| `soup data doctor` needs transformers | Only the light CLI is installed | `uv sync --group train` |
| Base-model download stalls at 0 B/s | Hugging Face Xet backend hanging | `export HF_HUB_DISABLE_XET=1` and rerun |
| `--output must stay under the current working directory` | `soup infer` path guard | Use a relative path like `data/predictions.jsonl` |
| Your `data.eval` file seems ignored | `DataConfig` has no eval-file field; the key is discarded | Set `val_split: 0` and evaluate after training ([Lesson 9](#l-splitting)) |
| Training slower than expected | Wrong backend | [Lesson 18](#l-qlora-backends) |

#### The debugging order

When a fine-tune misbehaves, check in this order — cheapest and most likely
first:

1. **`soup data doctor`** — is the rendered data correct? Especially `eos_in_labels`.
2. **`lr`** — is it ~2e-4 and not ~2e-5?
3. **Read 10 training rows** — do they actually look the way you intend?
   Include a few social ones; the frames pair randomly with answers and only a
   person can tell whether the join reads correctly.
4. **The loss curve** — did it move? Did it collapse to zero?
5. **`epochs`** — too few or too many?
6. **Response-shape coverage** — is the failing output format in your data at all?
7. **The base model** — is it capable of this task before any fine-tuning?

Step 7 is worth doing early when factual quality is the complaint. Run the
prompt against the untouched base model; if the base gets it wrong too, your
data is not the problem.

#### What this lesson teaches

Most fine-tuning bugs are data bugs or learning-rate bugs, and both are cheap to
check. Exhaust the seconds-long checks before spending another 15 minutes
training.

---

---

<a id="p8-shipping"></a>
## Part 8 — Shipping

---

<a id="l-shipping"></a>
### Lesson 23 — Merge, export, serve

> **Concept:** [D3 LoRA](FUNDAMENTALS.md#f-lora) · [D5 Precision and quantization](FUNDAMENTALS.md#f-quantization)

#### The idea

You have an 8.7 MB adapter. It needs its base model to run. This lesson covers
the four ways to ship it.

> Unlike the rest of the course, these commands were not executed here — each
> produces a multi-gigabyte artefact. Their flags are verified against
> `soup <cmd> --help`.

#### Option 1 — Keep the adapter separate

Smallest and most flexible. `soup serve` can host several adapters over one base
model, which is how you serve a different persona per route:

```bash
uv run soup serve --model ./output --port 8000
```

That exposes an **OpenAI-compatible** API, so existing clients work unchanged:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"output","messages":[{"role":"user","content":"What is entropy?"}]}'
```

Multiple personas, one base model in memory:

```bash
uv run soup serve --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapters ada=./output-ada \
  --adapters marie=./output-marie
```

Faster backends are available where installed:

```bash
uv run soup serve --model ./output --backend vllm --port 8000
```

#### Option 2 — Merge into a standalone model

Folds `B·A` back into `W`, producing ordinary weights with no adapter and no
inference-time overhead:

```bash
uv run soup merge \
  --adapter ./output \
  --base Qwen/Qwen2.5-1.5B-Instruct \
  --output ./merged
```

Trade-off: ~3 GB per persona instead of 8.7 MB. Merge when you want one
self-contained artefact; keep adapters separate when you want many personas.

#### Option 3 — Quantise for local use

GGUF runs in llama.cpp, LM Studio and Ollama:

```bash
uv run soup export --model ./merged --format gguf --quant q4_k_m --output ./gguf
```

`q4_k_m` is the usual quality/size sweet spot — roughly 1 GB for a 1.5B model.
Other formats: `onnx`, `tensorrt`, `awq`, `gptq`.

Straight into Ollama:

```bash
uv run soup deploy ollama --model ./merged --name ada-praise
ollama run ada-praise "What is the Pythagorean theorem?"
```

#### Option 4 — Publish

```bash
uv run soup push --model ./output --repo your-username/ada-praise-lora
```

Add `--private` to keep it unlisted. Generate a provenance-carrying model card:

```bash
uv run soup card <registry-id> -o MODELCARD.md
```

For anything regulated, Soup can emit a bill of materials:

```bash
uv run soup bom emit --name ada-praise --version 1.0 \
  --base-model Qwen/Qwen2.5-1.5B-Instruct --task sft \
  --license apache-2.0 --format both --output ./bom
```

#### Choosing

| Goal | Do this |
| --- | --- |
| Many personas, one server | Keep adapters separate, `soup serve --adapters` |
| One self-contained model | `soup merge` |
| Run on a laptop with no Python | `soup export --format gguf` |
| Share with others | `soup push` |
| Regulated environment | `soup bom emit` + `soup card` |

#### What this lesson teaches

The adapter is not the product — the adapter *plus a base model* is. Decide
early whether you are shipping many cheap personas or one convenient artefact,
because it changes how you package everything.

---

---

<a id="t-exercises"></a>
## Exercises

Each of these teaches something the lessons only described.

### 1. Change the person

```bash
uv run generate_data.py --name "Marie Curie"
uv run soup train --config soup.yaml
```

Nothing else needs editing. **Watch for:** whether domain-matched epithets still
feel right for a different figure, and whether a two-word name behaves like a
one-word name.

### 2. Widen the target modules

Set `target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]` in `soup.yaml` and
retrain, then re-run the tensor count from [Lesson 16](#l-adapter-file).

**Expected:** 224 tensors instead of 112, and roughly 4.36M parameters instead of
2.18M — the adapter grows from 8.7 MB to about 17 MB. Compare quality on the
shape probes. More capacity is not automatically better on 488 examples, and
this is the cheapest way to feel that trade-off.

### 3. Find your own coverage hole

Two axes to probe. For **output** shape, ask for a markdown table, a JSON object,
a numbered list, a haiku. For **input** shape, write prompts in a form the seed
set never uses — all-caps, multi-turn context, a pasted error message, two
questions in one line.

```bash
uv run soup infer --model ./output --input my_prompts.jsonl \
  --output data/my_predictions.jsonl --max-tokens 300
```

**Expected:** something degrades. On the output axis you will get prose where you
asked for structure; on the input axis you may lose the praise clause entirely.
Either way you have rediscovered [Lesson 6](#l-response-shape) on your own data.
Add seeds in that shape and confirm the fix.

Score it with the per-shape breakdown:

```bash
uv run check_persona.py --name "Ada Lovelace" \
  --predictions data/my_predictions.jsonl --probes my_prompts.jsonl
```

### 4. Overfit deliberately

Set `epochs: 10` and `lora.r: 64`, then retrain and ask a held-out question.

**Watch for:** loss falling much closer to zero, and answers that reproduce
training text nearly verbatim. Compare against the healthy 2.41 → 0.77 curve.

### 5. Add your own knowledge

Append real domain content to `seed_knowledge.jsonl`:

```json
{"domain": "cs", "question": "How do I reset my password?", "answer": "Open Settings, choose Security, then Reset password. A confirmation link is emailed to your registered address and expires after 30 minutes."}
```

For a brand-new domain key, add an entry to `EPITHETS` in `generate_data.py` or
let it fall through to `DEFAULT_EPITHETS`. Regenerate, re-run the doctor, retrain.

**This is the exercise that generalises to real work** — persona aside, the
pipeline is a working recipe for teaching a model your own house style.

---

---

<a id="t-glossary"></a>
## Glossary

The glossary lives in one place: **[FUNDAMENTALS.md#f-glossary](FUNDAMENTALS.md#f-glossary)**.
Every term set in **bold** or `code` in either file has an entry there.

---

<a id="t-next"></a>
## Where to go next

You now have a working pipeline. Things worth trying:

| Direction | Where to start |
| --- | --- |
| **Preference tuning** | `task: dpo` with `{prompt, chosen, rejected}` data. Teaches *which* answer is better, not just what a good answer looks like. |
| **A bigger base model** | `base: meta-llama/Llama-3.1-8B-Instruct` with `stream_layers: true` if VRAM is tight. Addresses the factual weaknesses in [Lesson 21](#l-case-study). |
| **Adapter arithmetic** | `soup adapters merge a b --strategy ties`, or `soup adapters arithmetic "coder + 0.5*math"`. Combine several LoRAs. |
| **Automated gating** | `soup train --gate <suite>` runs eval suites at epoch boundaries and refuses to ship a regression. |
| **Pre-flight in CI** | `soup ci init --data data/train.jsonl`. The doctor exits `2` on MAJOR, so it fails a build. |
| **Recipes** | `soup recipes list` — 142 pre-built configs across 23 training methods. |

Soup's own documentation is at [trysoup.dev/docs](https://trysoup.dev/docs).

---

---

<a id="t-summary"></a>
## Course summary

The eight things worth remembering:

1. **Fine-tune to change behaviour, use RAG to change knowledge.** Fine-tuning
   facts produces fluent, confident, wrong output.
2. **LoRA is the right tool for style, not a compromise.** 0.141% of parameters,
   an 8.7 MB artefact, laptop-trainable.
3. **Your JSONL is not what the model trains on.** Validate the *rendered* data.
   `eos_in_labels` above all.
4. **What you are training is a format, not a tone.** Hold the format constant
   and vary everything else — including register.
5. **Whatever shape is missing from your data gets erased**, on the output side
   *and* the input side. Prose-only data forgets how to write code; question-only
   data forgets how to handle `photosynthesis`.
6. **Split on the fact, not the row** — and check that your trainer does not
   re-split underneath you.
7. **Held-out metrics cannot tell a good model from a bad one.** All three
   rounds scored 100% there. Unseen topics and unseen input shapes found the
   failures.
8. **Let measurements set your config.** `max_length: 512` came from
   `truncation_risk`, not a guess.

The concepts behind all of it live in **[FUNDAMENTALS.md](FUNDAMENTALS.md)** —
go back to it whenever a number stops making sense.
