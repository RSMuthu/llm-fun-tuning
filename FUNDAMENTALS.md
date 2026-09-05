# Fine-Tuning from Zero — the concepts

This file explains the ideas. [TUTORIAL.md](TUTORIAL.md) makes you use them.

**There is not a single command here.** Every section ends with a link to the
lesson where you run the thing it describes. If you would rather start by typing,
begin at [Lesson 1](TUTORIAL.md#l-install) and follow the **Concept:** links back
here whenever a word stops making sense.

<a id="f-audience"></a>
## Who this is for

Someone who has used a chatbot and now wants to change how one behaves.

**Assumed:** none. No machine-learning background, no calculus, no GPU. Where
maths appears it is folded into a collapsible box you can skip without losing the
thread.

**You will need:** about 45 minutes, and a willingness to read "a matrix is a
grid of numbers" without flinching.

**What you get:** enough grounding that the tutorial's commands stop being
incantations. When it tells you to set `lora.r: 16`, you will know what rank is,
why 16, and what would break at 64.

<a id="f-how-to-read"></a>
## How to read this alongside the tutorial

| Route | Do this | Good if |
| --- | --- | --- |
| **Concepts first** | Read this file top to bottom, then start the tutorial | You like knowing why before how |
| **Hands first** | Start at [Lesson 1](TUTORIAL.md#l-install); follow each **Concept:** link back here when a term is unfamiliar | You learn by doing |
| **Reference** | Skip to the [glossary](#f-glossary) and search | You are already fluent and hit one unknown word |

All three work. The two files are written to be read in either order.

<a id="f-one-page"></a>
## The whole thing on one page

Two diagrams. Everything else in this file expands one of their boxes.

**How a model learns** — the loop that runs a few hundred times during training:

```text
   your text ──▶ tokens ──▶ [ model weights ] ──▶ probabilities for the next token
                    │                                        │
                    │                                        ▼
                    │                              compare with the token
                    │                              that actually came next
                    │                                        │
                    │                                        ▼
                    │                                    LOSS  (how wrong)
                    │                                        │
                    │                                        ▼
                    │                              GRADIENTS (which way to
                    │                              nudge each weight)
                    │                                        │
                    └──────── repeat ◀──── weights updated ◀─┘

   tokens: §A2   weights: §A3   loss: §B2   gradients: §B3   loop: §B1
```

**What LoRA changes** — why this fits on a laptop:

```text
   FULL FINE-TUNING                     LoRA
   ─────────────────                    ────
                                          W  ← frozen, untouched
    W  ← all 1.5B numbers               1,545,893,376 numbers
       updated                                  +
                                          B · A  ← trained
   needs ~24 GB                          2,179,072 numbers
                                         needs ~4.7 GB

   rank: §D2   LoRA: §D3   the adapter file: §D4   the memory: §D1
```

<a id="f-map"></a>
## Where to practise each concept

Every concept below is paired with the lesson that makes you do it.

> Maintenance note: this table also lives at [TUTORIAL.md#t-map](TUTORIAL.md#t-map).
> When a row changes, change it in both. Here the Concept links are bare `#f-…`
> and the Practice links are `TUTORIAL.md#l-…`; there they are reversed.

| Concept | Practise it |
| --- | --- |
| **A — What a language model is** | |
| [A1 What an LLM actually does](#f-what-is-an-llm) | [L3 Meet the base model](TUTORIAL.md#l-meet-the-base-model) |
| [A2 Tokens and tokenizers](#f-tokens) | [L4 Tokens on your own text](TUTORIAL.md#l-tokens) |
| [A3 Weights and parameters](#f-weights) | [L13 Estimating cost](TUTORIAL.md#l-cost-estimate) |
| [A4 Inside a transformer block](#f-transformer) | [L12 Rank, alpha and target modules](TUTORIAL.md#l-rank-alpha-modules) |
| **B — How a model learns** | |
| [B1 The training loop](#f-training-loop) | [L14 Your first training run](TUTORIAL.md#l-first-run) |
| [B2 Loss](#f-loss) | [L15 Reading the loss curve](TUTORIAL.md#l-loss-curve) |
| [B3 Gradient descent](#f-gradient-descent) | [L15 Reading the loss curve](TUTORIAL.md#l-loss-curve) |
| [B4 Batch, step, epoch](#f-batch-step-epoch) | [L11 Reading the config](TUTORIAL.md#l-config) |
| [B5 Learning versus memorising](#f-overfitting) | [L20 Held-out and out-of-domain](TUTORIAL.md#l-heldout-vs-ood) |
| **C — Kinds of training** | |
| [C1 Pretraining, SFT, alignment](#f-pretraining-sft-alignment) | [L11 Reading the config](TUTORIAL.md#l-config) |
| [C2 Prompting, RAG, or fine-tuning](#f-prompt-rag-finetune) | [L3 Meet the base model](TUTORIAL.md#l-meet-the-base-model) |
| **D — Making fine-tuning fit** | |
| [D1 Full fine-tuning, and why it does not fit](#f-full-finetuning) | [L13 Estimating cost](TUTORIAL.md#l-cost-estimate) |
| [D2 Rank, in actual linear algebra](#f-rank) | [L12 Rank, alpha and target modules](TUTORIAL.md#l-rank-alpha-modules) |
| [D3 LoRA](#f-lora) | [L12 Rank, alpha and target modules](TUTORIAL.md#l-rank-alpha-modules) |
| [D4 What is in an adapter file](#f-adapter-file) | [L16 Inside the adapter file](TUTORIAL.md#l-adapter-file) |
| [D5 Precision and quantization](#f-quantization) | [L17 Quantization in practice](TUTORIAL.md#l-quantization) |
| [D6 QLoRA](#f-qlora) | [L18 QLoRA, Unsloth and MLX](TUTORIAL.md#l-qlora-backends) |
| [D7 Choosing between them](#f-choosing) | [L18 QLoRA, Unsloth and MLX](TUTORIAL.md#l-qlora-backends) |
| **E — Using and judging the result** | |
| [E1 Inference](#f-inference) | [L19 Inference and temperature](TUTORIAL.md#l-inference) |
| [E2 What "it works" means](#f-evaluation) | [L20](TUTORIAL.md#l-heldout-vs-ood) · [L21](TUTORIAL.md#l-case-study) |
| **F — Hardware and memory** | |
| [F1 Where the memory goes](#f-memory-hardware) | [L1 Install](TUTORIAL.md#l-install) · [L22 Diagnosing failures](TUTORIAL.md#l-diagnosing) |

Tutorial Part 3 (dataset design, Lessons 6–10) has no concept section. That part
of fine-tuning is craft rather than theory, and the tutorial teaches it directly.

---

<a id="fa-what-a-model-is"></a>
## Part A — What a language model is

---

<a id="f-what-is-an-llm"></a>
### A1. What an LLM actually does

**In one sentence:** a large language model reads a sequence of text and predicts
which chunk of text is most likely to come next — and that is the whole of it.

Everything else is that one operation, run in a loop. To produce a sentence, the
model predicts one chunk, sticks it on the end of what it has, and predicts again
from the longer text. This is called being **autoregressive**: each output feeds
back in as input.

```text
   "The capital of France is"          ──▶  " Paris"   (most likely next)
   "The capital of France is Paris"    ──▶  "."
   "The capital of France is Paris."   ──▶  <end>
```

It is reasonable to find this unsatisfying. Answering questions, writing code and
refusing requests do not feel like next-chunk prediction. But if a model has read
enough text, then "the most likely continuation of a question is its answer"
turns out to carry an enormous amount of the work — and the later training stages
in [C1](#f-pretraining-sft-alignment) shape that raw tendency into something that
behaves like an assistant.

Two consequences worth carrying forward:

- **The model has no memory between conversations.** Everything it knows about
  the current exchange is in the text it is looking at right now, called the
  **context window**. Qwen2.5-1.5B's is 32,768 tokens.
- **It has no notion of truth**, only of likelihood. A fluent wrong answer and a
  fluent right one look equally plausible from the inside. This is why the
  tutorial keeps insisting that a persona fine-tune does not make a model more
  correct, only more consistent.

> **→ Practise it:** [Lesson 3 — Meet the base model before you change it](TUTORIAL.md#l-meet-the-base-model)

---

<a id="f-tokens"></a>
### A2. Tokens and tokenizers

**In one sentence:** a token is a chunk of text — usually a word-piece of about
four characters — and the tokenizer is the lookup table that converts text into
the integer token ids a model actually consumes.

Models do not see letters or words. Before anything else happens, your text is
split into tokens and each token is replaced by its id, a number indexing a fixed
**vocabulary**. Qwen2.5-1.5B's embedding table holds 151,936 rows. (A tokenizer
reports a couple of slightly different figures — `tok.vocab_size` is 151,643,
excluding tokens added after training, and `len(tok)` is 151,665. The embedding
size is the one that costs you memory.)

Splitting is learned from data, not from spaces. Common words are one token; rare
ones fracture (run this yourself in [Lesson 4](TUTORIAL.md#l-tokens)):

```text
   "What is photosynthesis?"

   "What"  " is"  " photos"  "ynthesis"  "?"
    3838    374      7249      73667      30
```

Note `" is"` carries its leading space — spaces belong to tokens. And note that
`photosynthesis`, a word you know perfectly well, arrives as two pieces. Names
fracture especially often, which is worth knowing for a project built on
repeating one person's name several hundred times.

**Special tokens** are entries in the vocabulary that mark structure rather than
text. The ones that matter here:

| Token | Purpose |
| --- | --- |
| **EOS** (end of sequence) | Marks where a reply stops. Qwen's id is 151645 |
| **BOS** (beginning of sequence) | Marks where text starts. Qwen does not use one |
| `<\|im_start\|>` / `<\|im_end\|>` | Qwen's chat markers, wrapping each conversation turn |

EOS earns its own warning. If the end-of-sequence token never appears in what the
model is trained to produce, it never learns that replies end, and at inference it
generates until it hits a hard limit. That is the single most common way a
fine-tune breaks, and the tutorial checks for it explicitly.

**Everything is measured in tokens, not words.** Sequence limits, memory, cost,
and speed all count tokens. A useful rule of thumb for English is ~4 characters
or ~0.75 words per token, but the only way to know is to run the tokenizer.

> **→ Practise it:** [Lesson 4 — Tokens on your own text](TUTORIAL.md#l-tokens)

---

<a id="f-weights"></a>
### A3. Weights, parameters, and what "1.5B" means

**In one sentence:** a parameter is a single number inside the model, and
training is the act of changing those numbers.

The numbers are organised into **matrices** — grids of numbers. A matrix's job is
to transform a list of numbers into another list of numbers. Feed in 1536 numbers
describing a token, multiply by a 1536×1536 matrix, get 1536 different numbers
out. Stack enough of these transformations and you get something that can
continue text.

A **weight** is one entry in one of those grids. "1.5 billion parameters" means
counting every entry in every matrix. For Qwen2.5-1.5B, the real shapes:

| Component | Shape | Parameters |
| --- | --- | --- |
| Hidden size | 1536 | the width of the model |
| Layers | 28 | identical blocks, stacked |
| `q_proj` (per layer) | 1536 → 1536 | 2,359,296 |
| `k_proj`, `v_proj` (per layer) | 1536 → 256 | 393,216 each |
| `o_proj` (per layer) | 1536 → 1536 | 2,359,296 |
| MLP `gate`/`up` (per layer) | 1536 → 8960 | 13,762,560 each |
| MLP `down` (per layer) | 8960 → 1536 | 13,762,560 |
| Vocabulary embedding | 151,936 × 1536 | 233,373,696 |
| Attention heads / KV heads | 12 / 2 | grouped-query attention, head dim 128 |

Add it all up and you land near 1.5 billion — the number in the model's name.

Two things follow. **Model size on disk is parameters × bytes-per-parameter**, so
the same model is 6.2 GB or 3.1 GB or 0.8 GB depending only on the number format
you store it in ([D5](#f-quantization)). And **training means changing these
numbers**, which is the entire difference between fine-tuning and prompting:
prompting changes what you feed the matrices, fine-tuning changes the matrices.

> **→ Practise it:** [Lesson 13 — Estimating cost before you pay it](TUTORIAL.md#l-cost-estimate)

---

<a id="f-transformer"></a>
### A4. Inside a transformer block

**In one sentence:** a transformer block lets every token look at every earlier
token and pull in what is relevant, then passes the result through a small
feed-forward network — and it is the names of the matrices doing the looking that
you will later choose between.

This section is scoped deliberately: enough to know what you are adapting, not
enough to implement one.

**Attention** is the looking-at-other-tokens part. For each token, the model
builds three vectors from three different matrices:

| Vector | Matrix | Role |
| --- | --- | --- |
| **Query** | `q_proj` | what this token is looking for |
| **Key** | `k_proj` | what each token offers as a match |
| **Value** | `v_proj` | what each token actually contributes |

Every token's query is compared against every earlier token's key. Strong matches
get high weight, and the token pulls back a blend of those tokens' values. A
fourth matrix, `o_proj`, mixes the result back into the main stream.

```text
   "The cat sat on the mat because it was tired"
                                       │
                        "it" forms a query: who am I referring to?
                        compares against every earlier key
                        strongest match: "cat"
                        pulls back "cat"'s value
```

That is why the names `q_proj` and `v_proj` matter to you. When
[LoRA](#f-lora) asks which matrices to adapt, `target_modules` names exactly
these. Adapting the query and value projections is the classic minimal choice —
it changes what the model looks for and what it retrieves, which turns out to be
where behavioural change lives.

After attention, an **MLP** (`gate_proj`, `up_proj`, `down_proj`) expands each
token's 1536 numbers to 8960, applies a non-linearity, and squeezes back to 1536.
**Residual connections** add each sub-layer's output back to its input, so
information can skip past, and **layer normalisation** keeps the numbers in a
sane range.

Then all of that repeats. 28 identical blocks, one after another.

> **→ Practise it:** [Lesson 12 — Rank, alpha and target modules in practice](TUTORIAL.md#l-rank-alpha-modules)

---
<a id="fb-how-learning-works"></a>
## Part B — How a model learns

---

<a id="f-training-loop"></a>
### B1. The training loop

**In one sentence:** training repeats six steps — take a batch of examples, run
them through the model, measure how wrong the output was, work out which
direction each weight should move, move them a little, repeat.

Every term in this part has a home in that loop:

```text
   1. BATCH        take a handful of training examples          §B4
   2. FORWARD      run them through the model, get predictions  §A1
   3. LOSS         measure how wrong the predictions were       §B2
   4. BACKWARD     work out which way to nudge each weight      §B3
   5. STEP         apply the nudges                             §B3
   6. repeat
```

Step 4 is **backpropagation**: the chain rule from calculus, applied backwards
through the network, producing for every single weight a number saying "if you
increase this, the loss goes up by roughly this much". You do not need to derive
it. You need to know it exists, that it is what the **backward pass** does, and
that it is why training needs far more memory than just running a model — every
intermediate value from the forward pass has to be kept around to compute the
backward one.

One pass over all your training data is an **epoch**. This project does three.

> **→ Practise it:** [Lesson 14 — Your first training run](TUTORIAL.md#l-first-run)

---

<a id="f-loss"></a>
### B2. Loss: what the number means

**In one sentence:** loss measures how surprised the model was by the token that
actually came next — lower means less surprised.

At each position the model outputs a probability for every token in its
vocabulary. Cross-entropy loss looks at whichever token actually followed and
takes the negative logarithm of the probability the model gave it. Confident and
right scores near zero. Confident and wrong scores enormously.

**This table is the most useful thing in this section.** It converts loss into
something you can picture:

| Loss | Probability on the correct token | Reading |
| --- | --- | --- |
| 4.61 | 1% | badly lost |
| 2.30 | 10% | weak |
| **2.20** | **11%** | where this project's training starts |
| 1.61 | 20% | learning |
| **1.03** | **36%** | where this project's training ends |
| 0.69 | 50% | strong |
| 0.10 | 90% | suspiciously strong on a small dataset |
| 0.01 | 99% | memorised |

Now "loss fell from 2.20 to 1.03" means something concrete: the model went from
putting about 11% of its confidence on the right next token to about 36%. And the
tutorial's warning that loss approaching zero on a few hundred examples is a
**red flag rather than a triumph** becomes obvious — 99% confidence on every
token of your training set means it has memorised the set, not learned the
pattern.

**Perplexity** is just `e` raised to the loss, so a loss of 1.03 is a perplexity
of about 2.8: the model is roughly as uncertain as if it were choosing between
2.8 equally likely tokens.

<details><summary>The maths, if you want it</summary>

For one position with correct token *t* and predicted probability *p(t)*:

```text
   loss = −log p(t)
```

Over a sequence, the reported loss is the mean of that across all positions
where a loss is computed. Note "where a loss is computed" — in chat fine-tuning
the user's tokens are usually excluded, so the number describes only how well
the model produces the *assistant* side.

</details>

> **→ Practise it:** [Lesson 15 — Reading the loss curve](TUTORIAL.md#l-loss-curve)

---

<a id="f-gradient-descent"></a>
### B3. Gradient descent

**In one sentence:** a gradient tells you which way to nudge a weight to make the
loss smaller, and gradient descent is nudging every weight that way, over and
over.

Picture the loss as a landscape. Every weight is an axis, so the landscape has
1.5 billion dimensions, but the idea survives the pictures we can draw: you are
standing somewhere on a hilly surface and want to reach the bottom. The
**gradient** is the slope under your feet. Descent means stepping downhill.

The **learning rate** is how big a step you take. It is the single most consequential
number in a fine-tuning config, and it fails in two directions:

```text
   TOO SMALL (lr = 2e-5 on a LoRA)      TOO BIG                    ABOUT RIGHT
   ────────────────────────────────     ───────                    ───────────
   ╲                                    ╲    ╱╲    ╱╲              ╲
    ╲___________                         ╲  ╱  ╲  ╱  ╲              ╲___
                                          ╲╱    ╲╱    → NaN             ╲______

   barely moves; your                   overshoots the valley       falls, then flattens
   fine-tune does nothing               and diverges
```

That left-hand picture is not hypothetical. `2e-5` is a sensible learning rate
for full fine-tuning and roughly ten times too small for LoRA, and the symptom is
a model that answers correctly with no trace of the behaviour you trained. The
tutorial has you reproduce it deliberately.

Two refinements you will see in the config:

- **Warmup** ramps the learning rate up over the first few percent of steps.
  Early gradients are large and erratic; stepping fully into them can wreck the
  adapter before it starts.
- A **scheduler** changes the learning rate over time. `cosine` decays it
  smoothly to near zero, so training takes big strides early and fine ones late.

The **optimizer** is what actually applies the update. Plain descent uses the
gradient directly; **Adam**, which nearly everything uses, keeps two running
averages per weight — roughly, recent direction and recent volatility — and uses
them to scale each weight's step individually. That is why it works well, and
also why it costs memory: two extra numbers for every trainable parameter, which
is where the 16-bytes-per-parameter figure in [D1](#f-full-finetuning) comes from.

<details><summary>The maths, if you want it</summary>

```text
   w  ←  w  −  lr × ∂L/∂w
```

Each weight *w* moves against its gradient, scaled by the learning rate. Adam
replaces the raw gradient with a normalised version built from its two running
averages, but the shape of the update is the same.

</details>

> **→ Practise it:** [Lesson 15 — Reading the loss curve](TUTORIAL.md#l-loss-curve)

---

<a id="f-batch-step-epoch"></a>
### B4. Batch, step, epoch, and the memory they cost

**In one sentence:** a batch is the handful of examples processed together, a
step is one update to the weights, and an epoch is one pass over all your data.

| Term | Meaning |
| --- | --- |
| **Batch** | Examples processed together in one forward pass |
| **Batch size** | How many. Bigger uses more memory and gives smoother gradients |
| **Step** | One weight update |
| **Epoch** | One full pass over the training set |

The complication is that big batches are good for stability and bad for memory.
**Gradient accumulation** resolves it: run several small batches, add up their
gradients, and only then take one step. You get the smoothness of a large batch
at the memory cost of a small one.

The number that matters is therefore not `batch_size` but the product:

```text
   effective batch = batch_size × grad_accum
```

With `batch_size: 1` and `grad_accum: 4`, the optimizer sees 4 examples per
step. From there the step count is arithmetic:

```text
   steps per epoch = training rows ÷ effective batch
   total steps     = steps per epoch × epochs
```

Two more terms that live here because they are memory, not maths:

- **Activations** are the intermediate values from the forward pass, held so the
  backward pass can use them. They scale with `batch_size × sequence length`,
  which is why sequence length is a memory knob and not just a truncation
  setting.
- **Gradient checkpointing** throws most activations away and recomputes them
  during the backward pass. Roughly 20% slower, dramatically less memory.

> **→ Practise it:** [Lesson 11 — Reading the config, knob by knob](TUTORIAL.md#l-config)

---

<a id="f-overfitting"></a>
### B5. Learning versus memorising

**In one sentence:** a model that has learned generalises to inputs it has never
seen; a model that has memorised only reproduces what it was shown, and loss
alone cannot tell you which you have.

This is the central risk of fine-tuning on a small dataset, and it is why the
loss table in [B2](#f-loss) matters. Training loss always falls. It falls whether
the model is extracting a pattern or building a lookup table, so a falling curve
is not evidence of success.

The only way to distinguish them is data the model has never been trained on.
Hold some out, and compare:

| Training loss | Held-out performance | Diagnosis |
| --- | --- | --- |
| falls | good | learning |
| falls to near zero | poor | memorising |
| barely moves | poor | undertrained, or learning rate too low |

Three levers control where you land:

- **Fewer epochs.** Every extra pass over a small set pushes toward memorisation.
- **Less capacity.** A smaller LoRA rank ([D2](#f-rank)) has less room to store
  specifics, so it is forced to generalise.
- **Dropout**, which randomly zeroes a fraction of values during training so the
  model cannot rely on any single pathway. `lora.dropout: 0.05` means 5%.

And one warning about held-out sets: they only work if the held-out examples are
genuinely unrelated to the training ones. If the same underlying fact appears on
both sides in different wording, your held-out score measures recall while
looking like generalisation. The tutorial spends a whole lesson on getting that
split right.

> **→ Practise it:** [Lesson 20 — Held-out, out-of-domain, and out-of-shape](TUTORIAL.md#l-heldout-vs-ood)

---
<a id="fc-kinds-of-training"></a>
## Part C — Kinds of training

---

<a id="f-pretraining-sft-alignment"></a>
### C1. Pretraining, SFT, and alignment

**In one sentence:** a chat model is built in three stages — pretraining gives it
knowledge and language, supervised fine-tuning teaches it to answer rather than
merely continue, and alignment polishes which answers it prefers.

| Stage | Data | What it learns | Scale | Output |
| --- | --- | --- | --- | --- |
| **Pretraining** | Raw text scraped at enormous scale | Language, facts, reasoning patterns | Trillions of tokens, millions of dollars | A **base** model |
| **SFT** | Prompt/response pairs written or curated by people | To respond in the shape of an answer | Thousands to millions of examples | An **instruct** model |
| **Alignment** | Pairs of responses, one preferred | Which of two valid answers is better | Thousands of comparisons | A finished chat model |

**Pretraining** is where all the knowledge comes from. A base model that has read
much of the internet knows the Pythagorean theorem — but ask it a question and it
may well continue with more questions, because that is a perfectly likely
continuation of a page containing a question.

**SFT — supervised fine-tuning** — fixes that. "Supervised" means every training
example carries the answer you want: you supply both sides, and the model is
trained to produce the response side. Mechanically it is still next-token
prediction, but only the assistant's tokens count toward the loss, so the model
learns to produce *that*.

**This is the stage this project operates in.** `task: sft` in the config, and
the training data is exactly prompt/response pairs.

**Alignment** learns from *comparisons* rather than examples: shown two answers
with one marked better, the model learns the preference. **DPO** (Direct
Preference Optimization) is the common method, needing `{prompt, chosen,
rejected}` triples rather than pairs.

### Base versus instruct, and why it matters here

The distinction decides which model you start from.

| | Base model | Instruct model |
| --- | --- | --- |
| Has been through | Pretraining | Pretraining + SFT (+ alignment) |
| Given a question | May continue it, list similar ones, or answer | Answers it |
| Named like | `Qwen2.5-1.5B` | `Qwen2.5-1.5B-**Instruct**` |

This project starts from **Qwen2.5-1.5B-Instruct**, and that choice does real
work. The goal is to change *style* while keeping normal answering behaviour
intact. Starting from the instruct model means the answering behaviour already
exists and only the style layer needs training. Starting from the base model
would mean teaching instruction-following and the persona at once, from a few
hundred examples — far harder, and the knowledge half would come out worse.

> A fourth possibility, **continued pretraining**, feeds more raw text to a model
> to teach it a domain's language. That is what a `text` data format is for. It
> is not what this project does.

> **→ Practise it:** [Lesson 11 — Reading the config, knob by knob](TUTORIAL.md#l-config)

---

<a id="f-prompt-rag-finetune"></a>
### C2. Prompting, RAG, or fine-tuning

**In one sentence:** prompting changes the instructions, RAG changes the facts
the model can see, and fine-tuning changes the model — and picking the wrong one
is the most expensive early mistake.

| Approach | Changes | Good for | Bad for |
| --- | --- | --- | --- |
| **Prompting** | Nothing; instructions sit in the context | Fast iteration, one-off behaviour | Consistency, long instructions, per-call cost |
| **RAG** | What facts are visible | Fresh or private knowledge, citations | Changing *how* the model writes |
| **Fine-tuning** | The weights | Style, format, tone, consistent behaviour | Teaching new facts reliably |

The rule worth memorising:

> **Fine-tune to change how a model behaves. Use RAG to change what it knows.**

Teaching facts by fine-tuning is possible and usually a mistake. The model does
not store your facts in a retrievable form; it adjusts its sense of what text is
likely. Feed it a hundred documents about your product and you get a model that
produces *plausible-sounding* product text, confidently, including details you
never wrote. There is no citation and no way to tell which parts are real.

Style is the opposite case. Style genuinely is a property of how the model
writes, it shows up in every response, and it is exactly what weight changes are
good at capturing. That is why this project is a fine-tune: the target is a
response *format*, not a body of knowledge.

> **→ Practise it:** [Lesson 3 — Meet the base model before you change it](TUTORIAL.md#l-meet-the-base-model)

---

<a id="fd-making-it-fit"></a>
## Part D — Making fine-tuning fit

This is the part that turns "you need a datacentre" into "you need a laptop".

---

<a id="f-full-finetuning"></a>
### D1. Full fine-tuning, and why it does not fit

**In one sentence:** updating every weight in a model requires roughly 16 bytes
per parameter, which for a 1.5-billion-parameter model is about 24 GB before you
have processed a single example.

The accounting, out loud. For every parameter you train, memory holds:

| What | Bytes (32-bit) |
| --- | --- |
| The weight itself | 4 |
| Its gradient | 4 |
| Adam's first moment | 4 |
| Adam's second moment | 4 |
| **Total per parameter** | **16** |

Multiply out:

| Model | Parameters | Full fine-tuning needs |
| --- | --- | --- |
| Qwen2.5-1.5B | 1.5B | **~24 GB** |
| Llama-3.1-8B | 8B | ~128 GB |
| Llama-3.1-70B | 70B | ~1,120 GB |

Mixed precision trims this to roughly 12–18 bytes per parameter, which changes
none of the conclusions. A 24 GB figure means a data-centre card for a model
small enough to run comfortably on a phone.

Full fine-tuning is not obsolete. When you are teaching genuinely new capability
rather than adjusting style, and you have the hardware, updating everything is
still the most direct route. It is simply the wrong tool for a response format.

> **→ Practise it:** [Lesson 13 — Estimating cost before you pay it](TUTORIAL.md#l-cost-estimate)

---

<a id="f-rank"></a>
### D2. Rank, in actual linear algebra

**In one sentence:** the rank of a matrix is the number of genuinely independent
directions it can produce — and a matrix can be enormous while its rank is tiny.

This is the idea LoRA is built on, so it is worth the five minutes.

A matrix maps a list of numbers to another list of numbers. Its **rank** counts
how many independent directions the output can span. Take this 3×3 matrix:

```text
    1   2   3
    2   4   6
    3   6   9
```

Nine numbers. But look: row 2 is row 1 doubled, row 3 is row 1 tripled. Every row
is a multiple of `[1, 2, 3]`. Whatever you feed in, the output lands somewhere
along a single direction. This matrix has **rank 1** — nine numbers' worth of
storage, one direction's worth of behaviour.

A rank-1 matrix can always be written as an **outer product**: one column vector
times one row vector.

```text
    ⎡1⎤                   1   2   3
    ⎢2⎥ × [1  2  3]  =    2   4   6
    ⎣3⎦                   3   6   9

    3 + 3 = 6 numbers      instead of 9
```

That generalises. **Any rank-r matrix is a sum of r outer products**, so a d×k
matrix of rank r needs only `r × (d + k)` numbers instead of `d × k`.

The saving grows with size. For a 1536×1536 matrix:

| Rank | Numbers needed | Versus full |
| --- | --- | --- |
| full (1536) | 2,359,296 | — |
| 64 | 196,608 | 12× smaller |
| **16** | **49,152** | **48× smaller** |
| 8 | 24,576 | 96× smaller |

Which sets up the claim everything else rests on:

> **The change a fine-tune makes to a weight matrix is empirically low-rank.**

A big matrix, a simple adjustment. Teaching a model to open every reply with a
praise clause does not require re-deriving language; it requires a small,
consistent nudge. And a small consistent nudge is precisely what a low-rank
matrix expresses.

> **→ Practise it:** [Lesson 12 — Rank, alpha and target modules in practice](TUTORIAL.md#l-rank-alpha-modules)

---

<a id="f-lora"></a>
### D3. LoRA

**In one sentence:** LoRA freezes the model and trains a pair of thin matrices
alongside each chosen weight matrix, so the update is stored in thousands of
numbers instead of millions.

Rather than learning a full update `ΔW` to add to a weight matrix `W`, LoRA
learns two skinny matrices whose product approximates it:

```text
              frozen                    trainable
        W  (d × k)          +        B (d × r) · A (r × k)

        2,359,296 numbers            49,152 numbers   (r = 16)
```

`W` never changes. Only `A` and `B` are trained. At inference their product is
added back, so there is no architectural change — and if you merge them into `W`
afterwards, no speed cost either.

**The arithmetic on this project's real matrices.** LoRA here adapts `q_proj` and
`v_proj` in each of Qwen's 28 layers:

| Matrix | Shape | Full | LoRA at r=16 | Saving |
| --- | --- | --- | --- | --- |
| `q_proj` | 1536 × 1536 | 2,359,296 | 16×1536 + 1536×16 = **49,152** | 48× |
| `v_proj` | 1536 × 256 | 393,216 | 16×1536 + 256×16 = **28,672** | 13.7× |
| per layer | | 2,752,512 | **77,824** | 35× |
| × 28 layers | | 77,070,336 | **2,179,072** | 35× |

Note that `v_proj` saves far less. LoRA's advantage comes from replacing `d × k`
with `r × (d + k)`, so it wins most on large square matrices and least on already-thin
ones. A 1536×256 matrix is not far off thin already.

**Alpha** scales how much the update counts. The adapter is applied as:

```text
   W + (alpha / r) × B·A
```

So `alpha` is a gain knob, and the `alpha = 2 × r` convention exists to keep that
gain constant as you change `r`. The trap: changing `r` while leaving `alpha`
fixed silently changes your effective learning rate, and you will misread the
result as a rank effect.

**B is initialised to zero.** So at step zero, `B·A` is zero, the adapted model
is bit-identical to the base, and training starts from a guaranteed no-op. There
is no moment where a randomly-initialised adapter is corrupting a good model.

**`target_modules`** names which matrices get adapters — the projections from
[A4](#f-transformer). Adapting `q_proj` and `v_proj` is the classic minimal
choice. Adapting more matrices means more capacity and a bigger adapter.

> **→ Practise it:** [Lesson 12 — Rank, alpha and target modules in practice](TUTORIAL.md#l-rank-alpha-modules)

---

<a id="f-adapter-file"></a>
### D4. What is actually in an adapter file

**In one sentence:** an adapter is two files — a small JSON recipe and a bundle
of the `A` and `B` matrices — and together they are a few megabytes against a
multi-gigabyte model.

**`adapter_config.json`** is the recipe for reattaching:

| Field | This project | Meaning |
| --- | --- | --- |
| `peft_type` | `LORA` | Which method |
| `r` | `16` | The rank from [D2](#f-rank) |
| `lora_alpha` | `32` | The gain from [D3](#f-lora) |
| `lora_dropout` | `0.05` | Dropout from [B5](#f-overfitting) |
| `target_modules` | `["q_proj", "v_proj"]` | Which matrices carry adapters |
| `base_model_name_or_path` | `Qwen/Qwen2.5-1.5B-Instruct` | What it attaches to |

**`adapter_model.safetensors`** holds the trained numbers — the `A` and `B`
matrices, named for where they attach:

```text
   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight   [16, 1536]
   base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight   [1536, 16]
   base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight   [16, 1536]
   base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight   [256, 16]
   ... and the same four for each of the 28 layers
```

28 layers × 2 matrices × 2 tensors = **112 tensors**. And the size follows
directly from [D3](#f-lora):

```text
   28 layers × (49,152 + 28,672)  =  2,179,072 numbers
   2,179,072 × 4 bytes (fp32)     =  8,716,288 bytes  ≈  8.7 MB
```

Which is the whole argument for LoRA in one line: **0.141% of the model's
parameters, 8.7 MB on disk, against a 3 GB base.**

**What is not in there** matters as much. No base weights, no vocabulary, no
tokenizer changes. An adapter alone cannot generate a single token — it is a
patch, and it needs the thing it patches. That is why sharing one means naming
the exact base model it was trained against.

> **→ Practise it:** [Lesson 16 — What is actually inside the adapter file](TUTORIAL.md#l-adapter-file)

---
<a id="f-quantization"></a>
### D5. Precision and quantization

**In one sentence:** quantization stores each weight in fewer bits, shrinking the
model in memory at a small cost in accuracy.

A number in memory is a bit layout, and you get to choose how many bits to spend.

| Format | Bits | Bytes/param | Qwen2.5-1.5B | Notes |
| --- | --- | --- | --- | --- |
| fp32 | 32 | 4 | 6.2 GB | Full precision. 1 sign, 8 exponent, 23 mantissa |
| fp16 | 16 | 2 | 3.1 GB | Half. Narrow exponent range, can overflow |
| **bf16** | 16 | 2 | 3.1 GB | fp32's exponent range, fp16's size. **Preferred for training** |
| int8 | 8 | 1 | 1.5 GB | |
| **4-bit** | 4 | 0.5 | **0.77 GB** | The `~0.8 GB` figure you will see in a memory profile |

bf16 versus fp16 is worth knowing: they are the same size, but bf16 keeps fp32's
exponent range and sacrifices mantissa bits. Training cares far more about range
than precision, because gradients span many orders of magnitude, so bf16 is the
safer default.

**NF4** — the 4-bit format used in practice — is cleverer than simply rounding.
Four bits give 16 possible values, and NF4 does not space them evenly. Weights in
a trained network are distributed roughly like a bell curve, clustered near zero,
so NF4 places its 16 levels where the weights actually are: densely near zero,
sparsely in the tails. Weights are quantized in small blocks with a scale factor
per block, and **double quantization** compresses those scale factors too.

**The mechanic everyone misses.** The weights are stored in 4 bits, but nothing
computes in 4 bits. During the forward pass each block is **dequantized** back to
bf16, the matrix multiply happens in bf16, and the 4-bit copy stays in memory.

Two consequences follow, and both surprise people:

- **Quantization buys memory, not speed.** It can even be slightly slower,
  because dequantizing is extra work. It is what makes a model *fit*.
- **Gradients flow to the adapter, never to the frozen base.** The base is
  read-only, so its precision only has to be good enough to compute a forward
  pass — which is exactly the opening [QLoRA](#f-qlora) walks through.

**Training-time and deployment-time quantization are different things**, and the
similar names cause real confusion:

| | Training-time | Deployment-time |
| --- | --- | --- |
| Looks like | `quantization: 4bit` in a training config | GGUF `q4_k_m`, AWQ, GPTQ |
| Purpose | Make the training run fit in memory | Make the finished model small and fast to serve |
| When | During training | After merging, on the way out |
| Needs | bitsandbytes + CUDA | Nothing special |

> **→ Practise it:** [Lesson 17 — Quantization in practice](TUTORIAL.md#l-quantization)

---

<a id="f-qlora"></a>
### D6. QLoRA

**In one sentence:** QLoRA is LoRA with the frozen base model quantized to 4-bit
— quantized base, bf16 adapters — and it is what lets large models be fine-tuned
on ordinary hardware.

That is the whole idea. Take [LoRA](#f-lora), and instead of holding the frozen
base in bf16, hold it in 4-bit [NF4](#f-quantization).

```text
   LoRA                            QLoRA
   ────                            ─────
   W  frozen, bf16   3.0 GB        W  frozen, NF4 4-bit   0.8 GB
   B·A  trained, bf16              B·A  trained, bf16
```

**Why it composes so neatly:** the base never receives gradients. Its only job is
the forward pass. So its precision only needs to be good enough to compute
activations, not good enough to accumulate tiny weight updates — and 4-bit turns
out to be good enough for the former and hopeless for the latter. The adapters
that *do* learn stay in bf16 where precision matters.

The QLoRA paper contributes three pieces: **NF4**, **double quantization**, and
**paged optimizers** (spilling optimizer state to CPU memory during spikes rather
than crashing).

**This repository ships a QLoRA config.** `soup.fast.yaml` sets a 4-bit
quantized base plus LoRA adapters at `r: 16` — quantized base, trained adapters,
which is QLoRA by definition, whatever the backend it runs on is called.

Honest caveats:

- A small quality cost against bf16 LoRA. Usually minor; not zero.
- Needs bitsandbytes and a CUDA-class GPU. It does not run on Apple Silicon or CPU.
- **The saving is on the base weights**, so it matters far more at 8B or 70B than
  at 1.5B. At this size the base is 3.0 GB and dropping it to 0.8 GB is pleasant;
  at 70B it is the difference between possible and impossible.

> **→ Practise it:** [Lesson 18 — QLoRA, Unsloth and MLX](TUTORIAL.md#l-qlora-backends)

---

<a id="f-choosing"></a>
### D7. Choosing: LoRA, QLoRA, or full fine-tuning

**In one sentence:** use LoRA for style and format work, QLoRA when the model
will not otherwise fit, full fine-tuning when you are teaching new capability and
own the hardware — and neither when a prompt would do.

| | Prompt / RAG | **LoRA** | **QLoRA** | Full fine-tuning |
| --- | --- | --- | --- | --- |
| What trains | nothing | thin `A`, `B` matrices | thin `A`, `B` matrices | every weight |
| Base precision | — | bf16 | **4-bit NF4** | bf16 |
| Memory, 1.5B | — | ~4.7 GB | ~2.5 GB | ~24 GB |
| Memory, 8B | — | ~18 GB | ~7 GB | ~128 GB |
| Artefact | — | ~8.7 MB | ~8.7 MB | full model, GB |
| Hardware floor | any | laptop GPU / Apple Silicon | CUDA-class GPU | data-centre GPU |
| Quality ceiling | n/a | high for style | slightly below LoRA | highest |
| Best for | one-off behaviour, fresh facts | **style, tone, format** | large models on small hardware | new capability |
| Used here | — | **yes, `soup.yaml`** | **yes, `soup.fast.yaml`** | no |

```text
   Do you need the model to behave differently, every time, consistently?
   ├── No  ──▶ prompt, or use RAG for facts
   └── Yes
       ├── Teaching genuinely new capability, with a big GPU?  ──▶ full fine-tuning
       └── Style, format, tone
           ├── Does the model fit in your memory in bf16?  ──▶ LoRA
           └── No, and you have a CUDA GPU             ──▶ QLoRA
```

You will also meet variants: **DoRA** splits the update into magnitude and
direction, **rsLoRA** rescales alpha for high ranks, and prefix and prompt tuning
train soft inputs instead of weight deltas. This project's `adapter_config.json`
exposes `use_dora` and `use_rslora` flags, both off. The concepts above transfer
directly.

> **→ Practise it:** [Lesson 18 — QLoRA, Unsloth and MLX](TUTORIAL.md#l-qlora-backends)

---

<a id="fe-using-and-judging"></a>
## Part E — Using and judging the result

---

<a id="f-inference"></a>
### E1. Inference: how text comes out

**In one sentence:** inference is running the model forward to generate text —
predicting a distribution over the vocabulary, picking one token, appending it,
and repeating until an EOS token or a limit.

Training changes weights; inference uses them. The loop:

```text
   1. run the prompt through the model
   2. get a probability for every one of 151,936 tokens
   3. pick one
   4. append it, go to 1
   5. stop at EOS, or at the token limit
```

Step 3 is where **temperature** lives. The model produces a distribution; the
temperature reshapes it before sampling.

| Temperature | Effect | Same prompt, three runs |
| --- | --- | --- |
| **0.0** | Always the highest-probability token | identical every time |
| **0.7** | Moderately flattened; likely tokens still favoured | varied but sensible |
| **1.5** | Heavily flattened; unlikely tokens get real chances | erratic, often incoherent |

Low temperature is repetitive and safe, high temperature is creative and unstable.
**top-p** (nucleus sampling) is the common companion knob: consider only the most
likely tokens whose probabilities sum to *p*, and ignore the long tail entirely.

There is a real consequence for evaluation: **anything measured at temperature
0.7 is a sample, not a measurement.** Run the same evaluation twice and the
numbers move. For a format-compliance score across twenty prompts that is
usually fine, but it is worth knowing before treating a percentage as exact.

**Two similarly-named limits that are not the same thing:**

| | Meaning | When |
| --- | --- | --- |
| `max_length` | Training truncation — rows longer than this get cut | Training |
| `max_tokens` | Generation cap — stop after this many new tokens | Inference |

And this is where [EOS](#f-tokens) collects its debt. A model that never learned
to emit EOS never stops on its own; it runs until `max_tokens` cuts it off, and
the symptom is a good answer followed by rambling.

> **→ Practise it:** [Lesson 19 — Inference: chat, infer, temperature, max tokens](TUTORIAL.md#l-inference)

---

<a id="f-evaluation"></a>
### E2. What "it works" actually means

**In one sentence:** a fine-tune works if it does the new thing on inputs it has
never seen, without having lost anything it could do before.

Three ideas; the tutorial does the work.

**Held-out is the floor, not the bar.** Testing on examples withheld from
training proves you did not memorise those rows. It does not prove the model
handles anything genuinely new, because held-out examples usually still resemble
training ones.

**A metric can be perfect while the output is worthless.** A score that checks
whether a required phrase appears will happily report 100% on responses that are
fluent nonsense. Read the actual outputs. Always.

**Ask what you removed.** Fine-tuning reshapes the whole output distribution, not
just the part you were aiming at. If every training example is a paragraph of
prose, the model learns that answers are paragraphs of prose — and quietly loses
the ability to emit a code block. The thing you broke is rarely the thing you
were testing.

> **→ Practise it:** [Lesson 20 — Held-out, out-of-domain, and out-of-shape](TUTORIAL.md#l-heldout-vs-ood) · [Lesson 21 — Case study](TUTORIAL.md#l-case-study)

---

<a id="ff-hardware"></a>
## Part F — Hardware, memory, and the words people use

---

<a id="f-memory-hardware"></a>
### F1. Where the memory goes

**In one sentence:** training memory is five things, each controlled by a
different setting, so running out of it is arithmetic rather than bad luck.

| Line | What it is | Set by | Shrink it with |
| --- | --- | --- | --- |
| **Model** | The frozen base weights | Model size × bytes/param ([A3](#f-weights), [D5](#f-quantization)) | Quantize to 4-bit, or a smaller model |
| **Adapter** | The trained `A`/`B` matrices | `lora.r`, `target_modules` ([D3](#f-lora)) | Rarely worth shrinking — it is megabytes |
| **Optimizer** | Adam's two moments, trainable params only | The optimizer ([B3](#f-gradient-descent)) | This is already the LoRA win |
| **Activations** | Forward-pass intermediates kept for the backward pass | `batch_size × max_length` ([B4](#f-batch-step-epoch)) | Lower either; enable gradient checkpointing |
| **Overhead** | Framework and driver context | Fixed | Nothing |

The **Optimizer** line is the one to stare at. Under full fine-tuning it would be
about 24 GB. With LoRA it is a rounding error, because Adam only tracks the 0.141%
of parameters that are actually training. That single row is the argument.

**The vocabulary you will meet in error messages:**

| Term | Meaning |
| --- | --- |
| **VRAM** | Memory on a discrete GPU. The hard limit on a CUDA machine |
| **Unified memory** | Apple Silicon shares one pool between CPU and GPU, so "VRAM" is whatever RAM is free |
| **CUDA** | NVIDIA's GPU compute platform. Most tooling assumes it |
| **MPS** | Metal Performance Shaders — PyTorch's Apple Silicon backend |
| **ROCm** | AMD's equivalent of CUDA |
| **OOM** | Out of memory. Add up the five rows above, compare to what you have |

> **→ Practise it:** [Lesson 1 — Install the toolchain](TUTORIAL.md#l-install) · [Lesson 22 — Diagnosing failures](TUTORIAL.md#l-diagnosing)

---
<a id="f-glossary"></a>
## Glossary

Every term that appears in **bold** or `code` in this file or in
[TUTORIAL.md](TUTORIAL.md) has an entry here. If you hit one that does not, that
is a bug worth reporting.

| Term | Definition | Explained in |
| --- | --- | --- |
| **Activations** | Intermediate values from the forward pass, kept so the backward pass can use them. Scale with `batch_size × max_length` | [B4](#f-batch-step-epoch) |
| **Adapter** | The small trained artefact from a LoRA run — the `A` and `B` matrices plus a config. Inert without its base model | [D4](#f-adapter-file) |
| **Alignment** | The third training stage, learning which of two valid answers is preferred | [C1](#f-pretraining-sft-alignment) |
| **alpha** | LoRA gain. The update is applied as `(alpha/r) × B·A`; convention is `alpha = 2r` | [D3](#f-lora) |
| **Attention** | The mechanism letting each token draw on earlier tokens, via query, key and value projections | [A4](#f-transformer) |
| **Autoregressive** | Generating one token at a time, feeding each output back in as input | [A1](#f-what-is-an-llm) |
| **Backpropagation** | The chain rule applied backwards through the network to get every weight's gradient | [B1](#f-training-loop) |
| **Base model** | A model that has had pretraining but not SFT. Also, loosely, whatever model an adapter attaches to | [C1](#f-pretraining-sft-alignment) |
| **Batch** | The examples processed together in one forward pass | [B4](#f-batch-step-epoch) |
| **bf16 / fp16 / fp32** | 16- and 32-bit float formats. bf16 keeps fp32's range at fp16's size, so it is preferred for training | [D5](#f-quantization) |
| **BOS / EOS** | Beginning- and end-of-sequence tokens. EOS in the training labels is what teaches a model to stop | [A2](#f-tokens) |
| **Chat template** | The Jinja template shipped with a tokenizer that turns a `messages` list into one token string. Model-specific | [L5](TUTORIAL.md#l-chat-template) |
| **Context window** | How much text the model can attend to at once. 32,768 tokens for Qwen2.5-1.5B | [A1](#f-what-is-an-llm) |
| **Continued pretraining** | Feeding more raw text to a model to teach it a domain's language | [C1](#f-pretraining-sft-alignment) |
| **Cross-entropy** | The loss function: the negative log of the probability assigned to the correct token | [B2](#f-loss) |
| **CUDA / MPS / ROCm** | GPU compute backends for NVIDIA, Apple Silicon and AMD | [F1](#f-memory-hardware) |
| **Dequantization** | Converting quantized weights back to bf16 at compute time. Why 4-bit saves memory, not time | [D5](#f-quantization) |
| **DoRA** | A LoRA variant splitting the update into magnitude and direction | [D7](#f-choosing) |
| **DPO** | Direct Preference Optimization — alignment from `{prompt, chosen, rejected}` triples | [C1](#f-pretraining-sft-alignment) |
| **Dropout** | Randomly zeroing a fraction of values during training so the model cannot lean on one pathway | [B5](#f-overfitting) |
| **Effective batch size** | `batch_size × grad_accum`. What the optimizer actually sees per step | [B4](#f-batch-step-epoch) |
| **Epoch** | One full pass over the training data | [B4](#f-batch-step-epoch) |
| **Forward / backward pass** | Running data through the model to get predictions; then computing gradients back through it | [B1](#f-training-loop) |
| **Full fine-tuning** | Updating every weight. ~16 bytes per parameter, ~24 GB for a 1.5B model | [D1](#f-full-finetuning) |
| **GGUF** | A deployment format for quantized models, used by llama.cpp, Ollama and LM Studio | [D5](#f-quantization) · [L23](TUTORIAL.md#l-shipping) |
| **Gradient** | For one weight, which direction and how steeply the loss changes if you nudge it | [B3](#f-gradient-descent) |
| **Gradient accumulation** | Summing gradients over several small batches before stepping. Large effective batch, small memory | [B4](#f-batch-step-epoch) |
| **Gradient checkpointing** | Discarding activations and recomputing them in the backward pass. ~20% slower, much less memory | [B4](#f-batch-step-epoch) |
| **Inference** | Running the model forward to generate text, as opposed to training it | [E1](#f-inference) |
| **Instruct model** | A base model that has been through SFT, so it answers rather than continues | [C1](#f-pretraining-sft-alignment) |
| **JSONL** | One JSON object per line. The standard training-data container | [L7](TUTORIAL.md#l-knowledge-base) |
| **Layer streaming** | Keeping frozen base layers out of VRAM so a large model trains on a small GPU | [L18](TUTORIAL.md#l-qlora-backends) |
| **Learning rate** | How large a step gradient descent takes. LoRA wants ~10× a full fine-tune's | [B3](#f-gradient-descent) |
| **LoRA** | Low-Rank Adaptation. Freeze `W`, train thin `A` and `B` where `ΔW ≈ B·A` | [D3](#f-lora) |
| **Loss** | How surprised the model was by the token that actually came next. Lower is better | [B2](#f-loss) |
| **Loss masking** | Excluding some tokens — usually the user's turn — from the loss, so only the reply is learned | [L5](TUTORIAL.md#l-chat-template) |
| **`max_length`** | Training-time truncation limit, in tokens | [E1](#f-inference) |
| **`max_tokens`** | Inference-time generation cap, in tokens. Not the same as `max_length` | [E1](#f-inference) |
| **Merging** | Folding `B·A` back into `W` to produce a standalone model with no adapter | [D3](#f-lora) · [L23](TUTORIAL.md#l-shipping) |
| **MLX** | Apple's array framework for Apple Silicon. One of Soup's three training backends | [L18](TUTORIAL.md#l-qlora-backends) |
| **NF4** | A 4-bit format whose 16 levels are spaced to match the bell-curve distribution of real weights | [D5](#f-quantization) |
| **OOM** | Out of memory. Add up the five memory lines and compare against what you have | [F1](#f-memory-hardware) |
| **Optimizer / Adam** | What applies the weight update. Adam keeps two running averages per parameter | [B3](#f-gradient-descent) |
| **Out-of-domain** | Prompts whose topic *or input shape* is absent from the training data. The real test | [E2](#f-evaluation) |
| **Outer product** | A column vector times a row vector, producing a rank-1 matrix | [D2](#f-rank) |
| **Overfitting** | Memorising training data instead of learning the pattern. Loss near zero on a small set is a warning | [B5](#f-overfitting) |
| **Parameter** | One number inside one of the model's matrices | [A3](#f-weights) |
| **Perplexity** | `e` raised to the loss. Roughly, how many equally likely tokens the model is choosing between | [B2](#f-loss) |
| **Pretraining** | The first and most expensive stage, learning language and facts from raw text | [C1](#f-pretraining-sft-alignment) |
| **QLoRA** | LoRA with the frozen base quantized to 4-bit NF4. Quantized base, bf16 adapters | [D6](#f-qlora) |
| **Quantization** | Storing weights in fewer bits to save memory | [D5](#f-quantization) |
| **`r` (rank)** | LoRA's capacity knob, and the linear-algebra rank of the update it can express. 8–16 for style, 32–64 for domain adaptation | [D2](#f-rank) · [D3](#f-lora) |
| **RAG** | Retrieval-Augmented Generation. Changes what a model can *see*, not how it behaves | [C2](#f-prompt-rag-finetune) |
| **Rank** | The number of genuinely independent directions a matrix can produce | [D2](#f-rank) |
| **Response shape** | The *form* of an answer — prose, code block, list, JSON. Shapes absent from training data get erased | [L6](TUTORIAL.md#l-response-shape) |
| **Scheduler** | How the learning rate changes over training. `cosine` decays it smoothly toward zero | [B3](#f-gradient-descent) |
| **SFT** | Supervised Fine-Tuning. Learning from prompt/response pairs, with loss on the response | [C1](#f-pretraining-sft-alignment) |
| **Step** | One update to the weights, consuming one effective batch | [B4](#f-batch-step-epoch) |
| **`target_modules`** | Which weight matrices get LoRA adapters. `q_proj` and `v_proj` here | [A4](#f-transformer) · [D3](#f-lora) |
| **Temperature** | How much the next-token distribution is flattened before sampling. 0 is deterministic | [E1](#f-inference) |
| **Token** | A chunk of text, typically ~4 characters, drawn from a fixed vocabulary | [A2](#f-tokens) |
| **Token accuracy** | The fraction of next tokens predicted exactly right | [B2](#f-loss) |
| **Tokenizer** | The component that converts text to token ids and back | [A2](#f-tokens) |
| **top-p** | Nucleus sampling: consider only the likeliest tokens summing to probability *p* | [E1](#f-inference) |
| **Transformer** | The architecture: stacked blocks of attention plus a feed-forward network | [A4](#f-transformer) |
| **Unsloth** | A CUDA-only accelerated training backend with fused LoRA kernels | [L18](TUTORIAL.md#l-qlora-backends) |
| **Vocabulary** | The fixed set of tokens a model knows. Qwen2.5-1.5B's embedding table has 151,936 rows | [A2](#f-tokens) |
| **VRAM / unified memory** | GPU memory. Apple Silicon shares one pool with the CPU | [F1](#f-memory-hardware) |
| **Warmup** | Ramping the learning rate up over the first steps so early gradients do not wreck the adapter | [B3](#f-gradient-descent) |
| **Weight** | One entry in one of the model's matrices. Synonym for parameter in practice | [A3](#f-weights) |

<a id="f-further-reading"></a>
## Further reading

- **LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., 2021. The
  original paper; readable, and short.
- **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023.
  Introduces NF4, double quantization and paged optimizers.
- **The Illustrated Transformer** — Jay Alammar. The standard visual explanation
  of attention, and the natural next step after [A4](#f-transformer).
- **Tokenizer playground** — any hosted tokenizer demo will let you paste text
  and watch it split, which makes [A2](#f-tokens) concrete in about a minute.
- **[trysoup.dev/docs](https://trysoup.dev/docs)** — the CLI this project drives.

---

Ready to build something? [TUTORIAL.md](TUTORIAL.md) starts at installation and
ends with a working fine-tuned model.
