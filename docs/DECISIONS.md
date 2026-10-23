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

**Superseded on 2026-10-22 by D13.** The runs move to a Colab T4 with an HF
token, so Gemma-2-2B-IT becomes the primary model and Qwen the secondary. No
Qwen2.5-1.5B data had been produced.

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

## D6 (2026-10-21) Who wrote the NoOp clauses

The 97 no-op clauses in `rfd.data.distractor.NOOP_CLAUSES` were written by the
AI assistant (Claude) in this session, one per usable template. Following the
GSM-NoOp recipe, each one is plausible but inconsequential, for example "{d} of
the pineapples picked on Thursday were slightly smaller than the rest". **No
human has reviewed them yet.** A clause that is not actually a no-op would make
some "distractor failures" correct answers to a changed question. The student
team should read all 97 before the results are cited (see the report's TODOs).
The distractor value d is drawn per instance (seeded) from 2-19. It excludes
every number in the question, every gold operand and result, and the unit
constants 7, 10 and 12, so any use of d in a model step is unambiguous.

## D7 (2026-10-21) Memory limits on the project Mac

On the 8 GB M2, Qwen2.5-1.5B (3.3 GB of fp16 weights on MPS) and the user's
other applications (about 5.5 GB) together overrun RAM. Swap traffic reached
about 260 MB/s and generation dropped from about 100 tok/s to under 20 tok/s.
`PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO` = 0.85/0.5 caps MPS growth. Batch 24 runs
out of memory under that cap, so batch 16 is the MPS default. Loading with
`device_map="mps"` segfaults with transformers 5.16 / torch 2.13, so we load
on CPU and then move the model.

## D8 (2026-10-21) Long runs move to another machine; local runs are debug only

Asked about the thrashing, the user chose to run the experiments on another
(GPU) machine. Until then:
- Code, tests and pre-registrations are completed on the Mac.
- The pipeline is exercised end to end on the Mac with **Qwen2.5-0.5B-Instruct**
  (`configs/debug.yaml`). Outputs go to `results/debug/`, which is gitignored,
  and **none of them are reported**.
- The harness is device-agnostic (`device: auto`; `dtype: auto` gives bf16 on
  CUDA and fp16 on MPS). Gemma-2 chat templates, which have no system role, are
  handled. If the GPU machine has an HF token with the Gemma-2 licence accepted,
  Gemma-2-2B-IT (the brief's primary model, and Sun et al.'s model) becomes
  feasible. Switching is a one-line change in `configs/harness.yaml` and needs
  re-running everything; we will not mix models within one report.

## D9 (2026-10-21) Choice of CoT prompt format

We compared three variants on 16 problems drawn at random from **all**
templates, using Qwen2.5-0.5B (scratch run, not kept):
- multi-turn demos: Final-Answer marker in 13/16 traces;
- multi-turn demos plus a format reminder in the last user turn: 16/16;
- everything in a single user message: 12/16.

Accuracy was 5-6/16 for all three. We chose the reminder variant **on marker
adherence, not accuracy**. Disclosure: the 16 problems were not restricted to
dev templates. The choice is about format only and was made with a different
model, so we judge the leakage into test results negligible, but it is a
deviation from "all selection on train/dev only".

## D10 (2026-10-21) Parser development check on 0.5B debug traces (not the pre-registered check)

To find parser bugs before the GPU run, the assistant inspected 30 sampled
steps from the **Qwen2.5-0.5B debug traces**
(`results/debug/traces/parse_sample.md`, not committed).

- **Extraction of (operation, operands, result) matched the step text in 28/30
  steps.** Both misses involve inline fractions:
  - `120 * 1/5` is read as `(120*1)/5`, so the op is labelled `/` instead of
    `*`;
  - `5/6 * 5/6` is evaluated left to right, so the value is right but the
    operands are odd.

  This is a known limitation and is not fixed.
- **Bugs found and fixed:**
  - parenthesised unit words (`74 (building blocks) + 46`);
  - bullet hyphens and earlier-sentence numbers leaking into the left-hand
    side;
  - fraction right-hand sides (`= 1/2`);
  - percent operands (`* 7%` read as 0.07) wrongly labelled "binding".

The pre-registered 30-step inspection (phase0.md §0.3) is still to be done on
the real Qwen2.5-1.5B traces.

## D11 (2026-10-21) Steering divergence is measured against an unsteered pass with identical batching

**The problem.** In the 0.5B debug run, removing the projection on a *random*
unit vector, which is a negligible edit, "diverged" 8/8 traces and moved net
accuracy by up to 25 points on 8 items. The cause was fp16 near-ties: argmax
under teacher forcing differs from argmax under batched KV-cached generation
for some tokens, regardless of the edit.

**The fix.** The engine now runs an unsteered teacher-forced pass on exactly
the same batch (cached per batch) and counts a position as changed only where
steered argmax ≠ unsteered argmax. Noise then cancels by construction, and the
zero-strength control should show 0 divergences.

**What remains.** Continuations of truly diverged traces are fresh generations
and carry the usual batch-level fp16 noise; this applies equally to real and
random directions. The exactness tests (fp32 tiny model) still pass. Baseline B
pre-registration Amendment 1 records this together with the widened additive
grid.

## D12 (2026-10-22) "Greedy" decoding was silently applying repetition_penalty = 1.1

**How it was found.** The Phase 0 check on the 0.5B debug model failed. Teacher
forcing reproduced only **94.1%** of the generated tokens, against the
pre-registered ≥ 0.99, and all 8 traces disagreed within their first 22 tokens.
fp32 gave **the identical 94.1% and the identical divergence positions**, so
this was not numerical noise.

**Cause.** The Qwen2.5-0.5B and 1.5B-Instruct checkpoints ship a
`generation_config.json` with `repetition_penalty: 1.1`, `top_k: 20`,
`top_p: 0.8` and `temperature: 0.7`. HF `generate()` applies the repetition
penalty even with `do_sample=False`, so the output was not the argmax of the
raw logits.

**Fix.**
- `rfd.harness.model.load` neutralizes the generation config.
- `rfd.harness.generate.generate` passes `repetition_penalty=1.0` explicitly.
- The T = 0.7 sampling in the pilot sets `top_p=1`, `top_k=0` and
  `repetition_penalty=1`.
- `tests/test_hooks.py::test_greedy_ignores_checkpoint_repetition_penalty`
  fails without the fix.

**Consequences.**
- Every 0.5B debug output produced before this fix was generated with the
  penalty. None of it is reported, and the debug traces must be regenerated
  before any further debug comparison.
- No Qwen2.5-1.5B data existed yet, so no reported result is affected.
- **The archived Semester-7 pilot was also affected**
  (`rfd.generation.generate_traces`, Qwen2.5-0.5B, 22/50 = 44% on GSM8K). It ran
  with repetition_penalty 1.1, not pure greedy decoding as `docs/PROGRESS.md`
  states. We leave that code and its results untouched for reproducibility and
  record the discrepancy here.

## D13 (2026-10-22) Primary model switched to Gemma-2-2B-IT on a Colab T4; Qwen2.5-1.5B becomes secondary

**Instruction (user, 2026-10-22).** Use `google/gemma-2-2b-it` as the primary
model. The target is a Colab T4: float32, `attn_implementation="eager"`, and
no system role in the chat template. `HF_TOKEN` comes from the environment.
Qwen stays as the secondary model.

**Config.**
- `configs/gemma2_2b.yaml` is the primary harness and the default of every
  experiment config and of the Makefile (`HARNESS`).
- The Qwen config moved from `configs/harness.yaml` to
  `configs/qwen2_5_1_5b.yaml`, unchanged apart from `dtype: auto`, which now
  gives float32 on GPUs without native bf16 (the T4 has none).
- A secondary run goes into its own results directory, for example
  `make all HARNESS=configs/qwen2_5_1_5b.yaml RESULTS=results/qwen2_5_1_5b`.
  `rfd.utils.resume.check_manifest` refuses to write a second model's outputs
  into a directory that already holds the first model's, so the report never
  mixes models.

**Why these settings.**
- **float32.** The T4 has no native bf16, and fp16 is unsafe for Gemma-2's
  activation range.
- **eager attention.** Gemma-2's attention-logit softcapping (50.0) is
  implemented only in the eager path.
- **No system role.** Gemma-2's template rejects a system turn. The harness
  config now says `system_role: false` and `prompts.chat_ids` folds the system
  text into the first user turn explicitly. Previously it fell back to this
  only after catching the template's exception.

**Gemma-2-specific fixes made for the switch.** None of these affected Qwen.
- Teacher-forced log-probs now apply Gemma-2's final logit softcap,
  30·tanh(z/30) (`harness.generate.lm_logits`). The bare `lm_head` omits it.
  The argmax is unaffected, because tanh is monotonic, but the token
  log-probabilities fed to Baseline B's uncertainty baselines would have been
  wrong.
- Phase 0 appended the gold answer with `add_special_tokens=True`, which for
  Gemma inserts a second `<bos>` mid-sequence. It now uses `False`. Phase 0 and
  the smoke test also check that every prompt carries exactly one BOS.
- On CUDA the model is loaded with `device_map="cuda"`. Shards go straight to
  the GPU, and the fp32 weights (≈ 10.5 GB) never sit in the 12 GB Colab host
  RAM.
- Batch sizes for the T4: generation 8, teacher-forced passes 4. The smoke
  test measures peak memory at the worst-case batch.

**Unchanged.** Data, splits, prompts, probes, steering grids and every
pre-registered threshold. The prompt format (D9) was chosen with Qwen-0.5B and
is used for Gemma as is, so Gemma's Final-Answer marker adherence is measured,
not tuned. Each pre-registration has a dated amendment recording the model
switch.

**Consequence.** Baseline A now runs on Sun et al.'s own model. The digit-length
rule is unchanged, and it may now pick 3 digits, the paper's setting.

## D14 (2026-10-22) Phase 0 re-run for Gemma-2: what has and has not been run

**Result on Gemma-2-2B-IT (2026-10-23, GCP L4, float32, eager attention;
`results/phase0/equivalence.json`): all six checks pass.**

| check | value | threshold |
|---|---|---|
| TL vs HF `resid_post`, median relative error, layers {0, 6, 13, 19, 25} | 1.47e-06 (worst layer) | < 0.01 |
| TL vs HF logits, top-1 agreement over 2083 positions | 1.0000 (max abs diff, last 32 positions: 8.0e-05) | ≥ 0.99 |
| padded batch vs single sequence, median relative error | 1.16e-06 (p99 2.6e-06) | < 0.01 |
| teacher-forced argmax vs batched greedy generation | 1.0000 over 990 tokens | ≥ 0.99 |
| effective generation config is pure greedy | do_sample False, repetition penalty 1.0, no temperature / top-p / top-k | as stated |
| one BOS per prompt | [1, 1, 1, 1] | 1 each |

Notes:
- Gemma-2's checkpoint ships no sampling defaults (all null; unlike Qwen's
  repetition penalty of 1.1, D12), so the neutralising code had nothing to
  remove. The check still guards against it.
- The TL build used the lossless bf16 weight source. Peak GPU memory was
  13.0 GB, of which TL's weights were 12.0 GB.
- Library versions: torch 2.9.1+cu129, transformers
  5.16.1, transformer-lens 3.9.0.
- The same VM passed `make smoke` (`results/smoke/`), measuring generation at
  124 tokens/s at the worst-case batch.

**What was run and passed.**

1. **Tiny random Gemma-2** (`tests/test_gemma2.py`; eager attention, softcaps
   on, bf16-representable weights; CPU, no download):
   - our HF decoder-layer hook vs TransformerLens `hook_resid_post` at every
     layer: max relative error < 1e-5;
   - TL vs HF logits: max absolute difference < 1e-4;
   - log-probs including the final softcap: error < 1e-4;
   - greedy generation with left padding is reproduced exactly by teacher
     forcing.

   A mutation check confirms the TL test fails when the float32 embedding fix
   (below) is removed.
2. **The rewritten Phase-0 script, end to end** on Qwen2.5-0.5B in fp32
   (`configs/debug_fp32.yaml`, local, **debug only, not reported**). This
   exercises the new low-memory TL path with a lossless bf16 weight source.
   - TL vs HF resid median relative error ≤ 3.7e-6 at layers {0, 6, 12, 18, 23};
   - logits top-1 agreement 1.0;
   - padding median relative error 1.4e-6;
   - greedy TF agreement 1.0 over 1442 tokens;
   - generation config: the checkpoint ships `do_sample: true`, T = 0.7,
     top-p 0.8, top-k 20 and repetition penalty 1.1; the effective config is
     pure greedy;
   - a single BOS in every prompt (Qwen has none).

**Phase-0 changes for the T4** (pre-registration amendment 1 in
`docs/prereg/phase0.md`; thresholds unchanged):
- Two checks were added:
  - the effective generation config must be pure greedy, with the
    checkpoint's own sampling defaults recorded (motivated by D12);
  - exactly one BOS per prompt.
- **Memory plan for the TL comparison.** Two fp32 copies (21 GB) fit neither
  the T4 nor the host RAM.
  - HF references are computed first. Only the argmax, plus full logits for the
    last 32 positions, are kept; a 256k vocabulary × 1000 positions would not
    fit in RAM.
  - The HF weights are then moved to the CPU as a bf16 copy. This is lossless
    because the checkpoint is stored in bf16 (5.23 GB for 2.6B parameters), and
    the script verifies it per tensor.
  - TL's float32 parameters are allocated directly on the GPU and filled from
    that copy.
  - TL scales Gemma's embedding by √2304 in the *source* dtype, which would round
    in bf16, so `W_E` is recomputed in float32 exactly as TL's fp32 path does.
  - **TL's logits are computed in position chunks.** This was added after the
    first Kaggle attempt ran out of memory, at 14.45 of 14.56 GiB on the T4.
    - TL does not tie `W_E` and `W_U`, so its fp32 Gemma takes about
      11.9 GiB, not 9.7.
    - `Unembed.forward` also builds `W_U.T.contiguous()`, a 2.2 GiB copy, on
      every call.
    - Phase 0 therefore runs TL with `return_type=None` and applies TL's own
      `ln_final`, `W_U`, `b_U` and `apply_softcap` to 64 positions at a time
      (`tl_logits`). It is the same arithmetic without the copy.
    - `tests/test_gemma2.py` checks that the chunked logits equal TL's
      built-in ones within 1e-5.
    - The 0.5B debug Phase 0 gives numbers identical to the previous run.

## D15 (2026-10-22) Every phase is resumable; results can live on Google Drive

**Results root.** Every `results/...` path in a config or on the command line
resolves under `$RFD_RESULTS_DIR`, which defaults to `results/`. The Makefile
sets it from `RESULTS=`. On Colab it points to a Drive folder, so outputs
survive runtime resets. `RFD_SCRATCH_DIR` (Colab local disk) receives the
assembled activation arrays, so Drive stores each array once, as shards.

**Restart semantics** (`rfd.utils.resume`).
- Every output file is written to a temporary file and renamed, so it is either
  whole or absent.
- A finished stage leaves `.done/<stage>.json`, which records a fingerprint of
  its input files. A stage is skipped only if its inputs are unchanged, and
  `--force` redoes it.
- Long stages save their unit of work as it finishes:
  - generation: chunks of 64 traces;
  - activations: shards of 64 traces / 256 arithmetic problems;
  - probe CV, Baseline-A probes and readability jobs: one cache file each;
  - steering and repair: every evaluated condition, including each of the
    100 random directions;
  - re-prompting and T = 0.7 sampling: per batch or item.

  Caches are keyed on the input fingerprint, so a redone upstream stage
  invalidates them.
- **Determinism.** A resumed run forms exactly the same batches as an
  uninterrupted one: the chunk plan is fixed and checked. The T = 0.7 samples
  are seeded per item. Results therefore do not depend on where a restart
  happened.
- **Tested** on the 0.5B debug model: `kill -9` during generation (after 2/6
  chunks) and during extraction (after 2/6 shards), then a restart. Traces,
  step labels and all three activation arrays were **byte-identical** to an
  uninterrupted run. Unit tests in `tests/test_resume.py`.
- `parse_sample.md` is never overwritten once it contains manual verdicts.

**Secrets.**
- The notebook reads `HF_TOKEN` from Colab's secret store into the process
  environment, and huggingface_hub reads it from there.
- No code path prints, logs or writes it. The smoke test records only whether it
  is set.
- `.env` is gitignored.

## D16 (2026-10-22) Kaggle variant of the GPU notebook

**Instruction (user).** Add `notebooks/run_on_kaggle.ipynb`:
- `HF_TOKEN` from `kaggle_secrets`;
- `RFD_RESULTS_DIR=/kaggle/working/results`;
- installation that never replaces the preinstalled CUDA torch;
- a CUDA assertion;
- restore from an attached dataset;
- a final zip of the results;
- safe under *Save & Run All*.

**Design.**
- **Install.** `pip list --format=freeze` for torch, torchvision, torchaudio,
  triton and nvidia-* is written to `/tmp/constraints.txt`. The pinned
  requirements, minus the `torch==` line, are installed with
  `-c /tmp/constraints.txt`. `pip list` is used rather than `pip freeze`
  because it always prints `name==version`, whereas `pip freeze` can print
  `name @ URL` lines, which pip rejects as constraints. If a pinned package
  needs a different torch, pip fails rather than replacing it. The next cell
  asserts `torch.cuda.is_available()` and that the torch version is unchanged.
- **No prompts, nothing silently skipped.** Setup cells raise on failure: a
  missing secret, a failed install, no CUDA, or a failing test or smoke check.
  This stops a Save & Run All before it burns GPU hours. Shell commands run
  through a helper that shows their output and raises on a non-zero exit;
  `!cmd` would not raise.
- **Deadline.** `rfd.utils.session.run_targets` runs the make targets in order,
  each in its own process group, until `T0 + 11.5 h − 25 min`. A target still
  running then is killed with SIGTERM, then SIGKILL. Every phase is resumable
  (D15), so the next session continues it. The first failing target stops the
  chain, because later phases need its outputs. This cell never raises, so the
  zip cell always runs.
- **Restore.** `restore_results('/kaggle/input', RESULTS)` finds `results/`
  folders, or `results*.zip` files when no folder is attached. It takes the one
  with the most completed-stage markers and copies only files missing from
  `RESULTS`. It never overwrites, so re-running it is harmless. The
  model-pinning manifest (D15) still refuses a different model's results.
- **Zip.** `results.zip` has `results/` as its archive root, so a dataset made
  from it restores directly. Above 8 GB the activation shards and caches are
  left out to stay under Kaggle's 20 GB output limit. The `results/` folder
  itself, which is in the output in full, is then the resume source.
- **Where things live.**
  - Code goes in `/tmp`, not `/kaggle/working`, so it is not part of the
    output.
  - Assembled activation arrays go in `RFD_SCRATCH_DIR=/tmp/rfd_scratch`.
  - Only `CUDA_VISIBLE_DEVICES=0` is used on a T4 x2.

**Tested.**
- `tests/test_session.py` covers:
  - restore from folders and from a zip, choosing the most complete one and
    keeping existing files;
  - stop at the first failure;
  - a deadline kill of a whole process group;
  - the zip's light mode.
- A local dry run executed the notebook's own cells with a stub
  `kaggle_secrets`, the local repo, and a scratch input holding a previous
  debug run. Restore copied 56 files, the phases ran under the deadline runner,
  and the zip and status files were written.
- The install and smoke cells need Kaggle's GPU image and were not run.

**Amendment (2026-10-22): restart safety.**
- **What happened.** On Kaggle the phases cell failed with
  `ModuleNotFoundError: No module named 'rfd'`, after Baseline A's collection
  had saved 4/22 shards. The kernel had restarted, and the cell depended on
  state from cells 1-6: `sys.path`, the environment variables, the deadline
  and the token.
- **Fix.**
  - Cell 1 now writes that state to `/tmp/rfd_session.py`, and every later
    cell re-runs it first. It covers paths, environment variables, the HF
    token from `kaggle_secrets`, `sys.path` and the working directory.
  - The session start time is kept in `/tmp/rfd_session_start`, so the
    deadline still counts from the start of the session.
  - The phase status is saved to `phase_status.json`, which the final cell
    reads if the kernel restarted in between.
  - If the code or install is gone, which means a new session, the cells say
    "use Run All" instead of failing obscurely. Run All skips finished work.
  - The CUDA check now runs in a subprocess, so the notebook kernel no longer
    holds a CUDA context (a few hundred MB of the T4) during the phases.
  - In the Colab notebook, the cells that run `make` now refuse to run without
    `RFD_RESULTS_DIR` and `HF_TOKEN`. Without the guard, a kernel restart would
    silently have written results to the clone's local `results/` instead of
    Drive.
- **Simulated locally:** a fresh process running only the phases and final
  cells works; a missing repo gives the "Run All" message.
- **Note.** In Kaggle's *interactive* editor, `/kaggle/working` survives a
  kernel restart but not the end of a session. Long runs belong in *Save & Run
  All*, whose output is kept.

## D17 (2026-10-22) Installing on preinstalled GPU images: wheels only, image's scientific stack kept

**What failed.** On Kaggle, the D16 install step stopped with
`Preparing metadata (pyproject.toml) did not run successfully`. The step was
`pip install -r requirements.txt` without torch, constrained to Kaggle's torch
family. The pasted log did not name the package, because `pip -q` hides the
"Collecting …" lines.

**Diagnosis (local, reproducible).**
- Checked every pin for prebuilt Linux wheels on Python 3.10-3.13.
  - On **Python 3.13**, `scipy==1.13.1`, `pandas==2.2.2` and
    `matplotlib==3.9.0` have no wheels, so pip compiles them from source.
  - `python-louvain==0.16`, which only the archived legacy code uses, has no
    wheel on any Python.
  - `transformers-stream-generator`, a transformer-lens dependency, is
    source-only but pure Python and builds fine.
  - `scikit-learn==1.9.0` needs Python ≥ 3.11.
- In a Python 3.13 venv that mimics a GPU image (torch 2.14.1 and the current
  numpy / scipy / pandas / matplotlib preinstalled), the old command fails with
  **the same message**. scipy's Meson build needs a Fortran compiler. So the
  Kaggle image's Python has no wheel for at least one pin. Which one cannot be
  certain from the truncated log, but the new install avoids every
  source-build path above.

**Fix.**
- `requirements.txt` no longer lists the legacy packages; they are in
  `requirements-legacy.txt`. `make install` installs both. `make test-fast`
  ignores `tests/legacy`. No current code imports a legacy-only package.
- `scripts/install_gpu_env.py`, used by both notebooks:
  - The image's torch, torchvision, torchaudio, triton and nvidia-* are frozen
    as constraints, as in D16.
  - It installs with `--only-binary=:all:`, so nothing is compiled; the
    exception is `transformers-stream-generator`.
  - The results-relevant libraries are installed at their **exact pins**:
    transformers, transformer-lens, datasets, huggingface-hub, tokenizers,
    safetensors, accelerate and scikit-learn.
  - The other pinned packages keep the image's preinstalled version when there
    is one. pip may still upgrade them, from wheels only, if a pin requires it.
  - It shows full pip output and writes `install_report.json` with every
    resulting version. The versions also go into each experiment's `versions`
    field, as before.
- **Deviation from "pinned versions everywhere".** On Kaggle and Colab,
  numpy / scipy / pandas / matplotlib / joblib / tqdm / PyYAML may differ from
  the pins, and torch is the image's (as in D16). These packages do not define
  the model or the probes. scipy's L-BFGS, which scikit-learn's logistic
  regression uses, can differ in the last digits across versions. The versions
  used are recorded with every result.

**Tested.**
- In the Python 3.13 venv, the script installs cleanly:
  - torch stays 2.14.1, and numpy / scipy / pandas keep their image versions;
  - transformers 5.16.1, transformer-lens 3.9.0 and scikit-learn 1.9.0 are
    exact;
  - all 77 current tests pass in that environment.
- `tests/test_install_gpu_env.py` covers the install plan and the
  requirements split.
- Not yet run on Kaggle itself.

## D18 (2026-10-23) Runs move from Kaggle to Google Cloud (project `icl-failure`, one L4)

**Instruction (user).** Stop working on Kaggle; use the GCP account and the
project `icl-failure`, and start the work there.

**What was set up in the project.**
- Defaults: `gcloud config set project icl-failure` (the account is
  dahiya24mehul@gmail.com; before the change, the default was `research-507316`).
- APIs enabled: Compute Engine, Cloud Quotas, Secret Manager, IAM.
- **GPU quota.** `GPUS_ALL_REGIONS` was **0**, which blocks every GPU even
  though the per-region L4, T4 and V100 quotas were 1. I requested **1** through
  the Cloud Quotas API (preference `gpus-all-regions-1`), and it was granted
  immediately. The first attempt failed with HTTP 429, because gcloud charged it
  to its own shared project; it needs `--billing-project=icl-failure`.
- Bucket `gs://icl-failure-rfd-results` (us-central1) for results and status.
- Service account `rfd-runner`, with only:
  - `storage.objectAdmin` on that bucket;
  - `secretmanager.secretAccessor` on the secret `hf-token`;
  - `logging.logWriter`.
- The secret `hf-token` was created **empty**. Only the user adds the value.
  The token is not stored in any file, the metadata or the repo.

**Why an L4 and not a T4.**
- fp32 FLOPs: L4 30 TFLOPS vs T4 8.1.
- Memory: 24 GB vs 15 GB, so the fp32 Gemma (10.5 GB) leaves room for 3x larger
  generation batches. Decoding is memory-bandwidth-bound, so throughput scales
  with batch size.
- TransformerLens's untied fp32 copy (11.9 GiB, D14) fits comfortably.
- An L4 hour costs more than a T4 hour, but fewer hours are needed.

**Config.** `configs/gemma2_2b_l4.yaml` has the same model settings as the T4
config: float32, eager attention, `system_role: false`. Only the batch sizes
differ (generation 24 instead of 8). `run_manifest.json` checks model, dtype and
attention, so the two configs' result directories are interchangeable. Batch
size is a hardware setting, not an experimental one. Greedy decoding with
left-padding is invariant to it up to fp32 rounding, which Phase 0 checks
(padding invariance ≤ 1e-2, greedy reproducibility ≥ 0.99). `make smoke` runs
the worst-case batches before any long run, and fails on an out-of-memory.

**Operation** (`scripts/gcp/`).
- `launch.sh`: an idempotent create-or-start of the VM `rfd-gemma`
  (g2-standard-8, Deep Learning pytorch/CUDA 12.9 image, 200 GB disk).
- The VM's startup script clones the repo and starts `run_phases.sh` as a
  detached systemd unit on every boot, so a reboot resumes the run.
- `run_phases.sh`:
  - installs with `scripts/install_gpu_env.py` into a venv that sees the
    image's CUDA torch (D17);
  - runs the unit tests, then waits up to 45 minutes for the `hf-token` secret;
  - runs the smoke test, then every phase in order;
  - syncs results to the bucket every 10 minutes and after each phase;
  - writes `status.json` to the bucket;
  - shuts the VM down when finished, on a failure, or when the token never
    arrives, so a failed or idle run does not keep billing.
- A fresh disk restores from the bucket first.

**What carries over.** The Kaggle session produced nothing that is reported.
Its `equivalence.json` was never retrieved, so Phase 0 is run again here and is
the first result.

## D19 (2026-10-23) Reduced-scale profile, because of a time limit

**Instruction (user).** There is little time before the presentation to the
supervisor. Shrink the work so that some baseline results can be shown, in about 6-8
hours instead of 14-18 at best.

**Why the full run is so long.** From the measured speeds, almost all of it is
Baseline B's steering: about 410 conditions (strength selection, three directions
at the selected strength, 3 x 100 random directions, and a descriptive sweep), each
needing a teacher-forced pass over 128-160 traces plus regeneration of those that
diverge. Trace generation and the CPU probe fits are small by comparison.

**What was cut** (amendments dated 2026-10-23, written before any data of the affected
stages existed; sizes in `configs/reduced/`):
- Traces: 4 instances per template instead of 8 (Phase 0 amendment 2).
- Baseline B: 32 + 32 selection traces and 40 + 40 test traces instead of
  64 + 64 and 80 + 80; **20 random directions per condition instead of 100**; 2 projection
  strengths and 4 additive strengths instead of 5 and 7; a 3-direction sweep band
  (amendment 3).
- Pilot: 20 random directions instead of 100 (amendment 2).

**What was not cut.** Every threshold and criterion (AUROC / balanced accuracy
≥ 0.75 above shuffled, 20% gold-value decodability, 5-point minimal effect); the
template-grouped splits, template-clustered bootstrap with 10,000 resamples, train-only
selection, greedy decoding, the model and its precision. **Baseline A is unchanged**: it
had already started, and it is cheap (about 1.5 hours).

**What it costs scientifically, and how it is kept honest.**
- The brief's "≥ 100 random directions per steering condition" requirement is not
  met, and the 5-point minimal effect cannot be resolved with 80 test traces (intervals
  about ±10 points). Results can show a large effect or rule one out, not decide a
  small one.
- `make report` compares the run's actual scale with the pre-registered scale
  (`report.PREREG_SCALE`, read from the result files) and prints a "Scale vs
  pre-registration" table. It marks every verdict that depends on a reduced quantity
  **provisional**. No amount of editing the prose can remove that mark, because it
  comes from the data. Each stage also writes `config_used.json`.
- The reduced profile is selected with an untracked `local.mk` (`PROFILE := reduced`),
  so the repo default stays the full pre-registered scale.
- **Extension.** Conditions are cached per direction, so the 20 random directions
  are the start of the 100: `baseline_b steer --force --n-random 100` computes only the
  missing 80 per direction type, and the pilot is the same. The larger evaluation sets
  and instance counts would need a full rerun. Results of the reduced profile can be
  presented as "baseline replication at pilot scale".

**Estimate** (not a measurement): Baseline A about 1 h more, traces 0.7 h, Baseline B
probes 0.1 h, steering about 2.5 h, pilot about 0.7 h, so 5-6 h in total.

## D20 (2026-10-23) Exploratory check of the pilot's repair edit (not a pre-registered experiment)

**Why.** The pilot's repair test reported "items whose continuation changed: 0.00" in every
condition, including the oracle residual patch and the strengths up to 8×. That is
either a real floor effect or a bug that stops the edit from being applied. A
bug would have made the pilot's headline number meaningless, so it was checked before any
interpretation.

**What was run.** The same repair code (`pilot repair`), the same probes and class
means, at one strength multiple of 64 and one of 512 (`configs/exploratory/`), 2 random
directions, into separate directories (`results/pilot_diag_64`, `results/pilot_diag_512`).
The pilot's own `results/pilot/` is untouched.

**Result.**
- The edit is applied. At 64× and 512× it changes the continuation of 68% (oracle
  content), 59% (self-sourced gated) and 100% (evaluation) of the items. The pre-registered
  multiples {1, 2, 4, 8} are far too weak for a single-token edit at the first token of a step.
  The oracle residual patch (replace one position's residual with the matched correct run's)
  changes 0 items, because the two residuals at that token are nearly identical.
- The oracle content edit gives net +22.7 points [5.0, 42.1], but random directions of the same
  norm give +18.2 (512×) and +15.9 (64×) (means of 2), and evaluation steering gives +13.6
  [0.0, 26.1]. So large perturbations repair some failures regardless of direction, and there is
  no sign of a content-specific effect. With 15 failures and 2 random directions, this is
  not a test either.

**Status.** Exploratory: not part of any verdict, not pre-registered, and not a
substitute for the pilot. It supports calling RQ2 "inconclusive" rather than "refuted", and
motivates the redesign in report §4 (edits across the step, calibrated norms, more failures,
≥ 100 random directions).

**Operational note.** The exploratory run restarted the finished VM. Its auto-runner had to
be stopped by hand, and the VM then ran idle for about 2.5 hours before being stopped.
`rfd-gemma` is now TERMINATED (disk cost only).
