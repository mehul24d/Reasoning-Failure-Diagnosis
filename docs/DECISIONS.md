# Decision log

Decisions taken without asking, with the reason for each. Newest last.
Dates are when the decision was made.

## D1 (2026-10-21) Model: Qwen2.5-1.5B-Instruct instead of Gemma-2-2B-IT

The brief names Gemma-2-2B-IT as the primary model, with Qwen2.5-1.5B-Instruct
as the fallback. **We use the fallback.** Reasons, all checked on this machine:

- No `HF_TOKEN` is configured (`~/.cache/huggingface/token` is absent, `HF_TOKEN`
  is unset), and Gemma-2 is a gated repository, so it cannot be downloaded.
- Even with a token, it would not be feasible: the machine is an Apple M2 with
  **8 GB unified memory and no CUDA GPU** (PyTorch MPS backend only). Gemma-2-2B
  has 2.6B parameters (about 5.2 GB in fp16). The OS was already using 5-8 GB of
  swap while we benchmarked the 1.5B model, and the experiments need thousands
  of greedy generations plus steering sweeps.
- Qwen2.5-1.5B-Instruct is already cached locally, its tokenizer splits numbers
  into single digits (as Gemma's does, which Sun et al. rely on), and it is one
  of the models in Yuan et al. (2605.09502).

Consequence: none of the numbers here are Gemma numbers. Sun et al. used
Gemma-2-2B-IT, so Baseline A is a cross-model replication, not an exact one.

## D2 (2026-10-21) Gold programs come from the GSM-Symbolic *templates*

The brief asks us to extract gold intermediate values from GSM8K-style
`<<a op b = c>>` annotations in the gold solution. **The HF dataset
`apple/GSM-Symbolic` has none**: 0/5000 `answer` and 0/5000 `original_answer`
fields contain `<<`. The template files in the GitHub repository
`apple/ml-gsm-symbolic` (`templates/symbolic/*.json`) do contain them: there is
an `answer_annotated` field with `<<{expr}={expr}>>` slots, and a
`question_annotated` field with `{var,default}` slots. We match each instance
question against its template to recover the variable values, then render
`answer_annotated` with those values. We count a reconstruction as valid only if
the rendered text, with the annotations stripped, equals the HF `answer`
exactly. Instances that fail this check are dropped, and the drop rate is
reported.

## D3 (2026-10-21) GSM-NoOp is not released, so we build distractor variants

Neither HF (`main`, `p1`, `p2` only) nor the GitHub repository (`p1`, `p2`,
`symbolic`) contains GSM-NoOp. Cox et al. (2603.01437) also note that it was
never released and re-created it with LLM-proposed clauses. We insert a no-op
clause containing a number into each problem, keeping the distractor quantity as
an explicit slot (see `rfd.data.distractor`).

## D4 (2026-10-21) Legacy code layout

`rfd.{sae,segmentation,taxonomy,metrics}` moved to `rfd.legacy.*`. Configs moved
to `configs/legacy/`, the run script to `scripts/legacy/`, and the tests to
`tests/legacy/`, with a `legacy` pytest marker. The old `make pilot` is now
`make legacy-pilot`, and its outputs moved to `results/legacy_pilot/`, so the
new `make pilot` (Phase 3) does not overwrite them.

## D5 (2026-10-21) Where the papers differ from the brief

We fetched all four papers (arXiv HTML). They are reachable, but no code
repository is linked for Sun et al. or Yuan et al. Differences that change what
we implement:

- **Sun et al. (2507.12379).** Model: Gemma-2-2B-IT. Task: 3-digit addition
  (sum < 1000), 2-shot, answer format `<<x+y=z>>`. Probes read the residual
  stream at the `=` token at every layer. Targets: the hundreds digit of the
  model's output and of the ground-truth sum, the only digit where Gemma
  erred. Probe types: linear, logistic, 1-hidden-layer MLP (512), and circular.
  - **Probes are trained on a mixed set that includes erroneous cases**:
    100 per predicted-hundreds-digit class, 70/30 split. The brief's protocol
    is different: train on correct cases, evaluate on erroneous ones. We run
    both and label which is which.
  - The paper **reports no accuracy restricted to erroneous cases** and **no
    shuffled-label or random-probe controls**. We add both.
  - Selective re-prompting: an MLP error detector at layer 25 flags steps in
    *GSM8K CoT*. It reports TP-correction and FP-preservation only, **not task
    accuracy versus no intervention or versus always re-prompting**. We do the
    comparison the brief asks for, in the arithmetic setting.
- **Yuan et al. (2605.09502).** Data: MATH-500 (100 train / 200 eval
  problems), **not GSM-Symbolic**. Probe: standardized logistic regression
  (C=0.1) at the **last token of the full trace**, layer chosen by 5-fold CV.
  Baselines: TF-IDF text classifier, verbalized confidence, sequence log-prob,
  self-consistency, CCS, P(True). **No question-only baseline.**
  - Steering is **projection removal** along the normalized probe weights,
    `h' = h - a (h.w) w` with a in {0.5, 1, 2, 5, 8}, applied during generation.
    The brief instead describes adding a difference-in-means or probe-weight
    direction. The paper does not state the steering layer or positions, and
    **has no random-direction controls**. We run the paper's operation and the
    brief's additive variant, each with matched-norm random controls.
  - Prompt format: `Step k:` ... `ANSWER:`. We use the brief's
    `Step k:` ... `Final Answer:`.
- **Cox et al. (2603.01437).** Probe: difference-in-means at the last token
  before the CoT, cosine score. Steering adds `a*w` at every decoding
  position. The null is one norm-matched orthogonal direction per example.
  We use their norm-matched random-direction idea but draw >= 100 directions
  per condition, as the brief requires.
- **Qu & Gomes (2609.17804).** Model: Llama-3.3-70B. Distractor failures are
  localized to an Operation Planning stage (L23-L36) by patching
  Filler-correct into NoOp-wrong runs. Head scaling (alpha=2) on an H- set
  recovers about half of NoOp failures. They do not build a targeted repair
  method. Code: github.com/deliaqu/llm-reasoning-decomposed. This motivates
  our choice of variables (chosen operation, distractor bound) and of
  distractor failures as the pilot population.
