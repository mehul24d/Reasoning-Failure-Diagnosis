"""Prompt templates (plan Appendix G) and chat-template wrapping.

Two styles, three paraphrases each. Paraphrase 0 of each style is the wording printed in the
plan; paraphrases 1 and 2 are DRAFTS and must be reviewed by the group before they are frozen
at M0 (the plan fixes the final templates in the repository at M0).

Tokenisation rule: the chat template already inserts the BOS token for Gemma, so the rendered
string is tokenised with add_special_tokens=False. Tokenising it with special tokens added
produces a duplicate BOS, one of the four issues found in the preliminary run (Section 3.9.1).
"""
from __future__ import annotations

from dataclasses import dataclass

MARKED = (
    'Solve the following problem step by step. Write each step on a new line in the form '
    '"Step k: <calculation> = <result>". End with "Answer: <number>".\n'
    "Problem: {problem}",
    # DRAFT paraphrase 1
    'Work through the problem below one step at a time. Put each step on its own line as '
    '"Step k: <calculation> = <result>". Finish with "Answer: <number>".\n'
    "Problem: {problem}",
    # DRAFT paraphrase 2
    'Solve this problem in numbered steps. Each line should read "Step k: <calculation> = '
    '<result>". On the last line write "Answer: <number>".\n'
    "Problem: {problem}",
)

TERSE = (
    'Solve the problem. Show brief working, then give the final answer after "Answer:".\n'
    "Problem: {problem}",
    # DRAFT paraphrase 1
    'Solve the problem below. Keep your working short and give the final answer after "Answer:".\n'
    "Problem: {problem}",
    # DRAFT paraphrase 2
    'Answer the following problem. Briefly show your working, then write "Answer:" followed by '
    "the final answer.\n"
    "Problem: {problem}",
)

TEMPLATES: dict[str, tuple[str, ...]] = {"marked": MARKED, "terse": TERSE}


@dataclass(frozen=True)
class PromptSpec:
    style: str
    paraphrase: int

    @property
    def tag(self) -> str:
        return f"{self.style}_p{self.paraphrase}"


def all_prompt_specs() -> list[PromptSpec]:
    return [PromptSpec(s, p) for s, ts in TEMPLATES.items() for p in range(len(ts))]


def render_user_message(problem: str, spec: PromptSpec) -> str:
    try:
        template = TEMPLATES[spec.style][spec.paraphrase]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unknown prompt spec {spec}") from e
    return template.format(problem=problem.strip())


def render_chat(tokenizer, problem: str, spec: PromptSpec) -> str:
    """Wrap the user message in the model's own chat template, ready for generation."""
    messages = [{"role": "user", "content": render_user_message(problem, spec)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_chat(tokenizer, problem: str, spec: PromptSpec) -> list[int]:
    """Token ids of the chat-wrapped prompt, with exactly one BOS where the model uses one."""
    text = render_chat(tokenizer, problem, spec)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert_single_bos(tokenizer, ids)
    return ids


def count_bos(tokenizer, ids: list[int]) -> int:
    bos = tokenizer.bos_token_id
    if bos is None:
        return 0
    return sum(1 for t in ids if t == bos)


def assert_single_bos(tokenizer, ids: list[int]) -> None:
    """Gemma: exactly one BOS, at position 0. Models without a BOS token (Qwen): none."""
    n = count_bos(tokenizer, ids)
    if tokenizer.bos_token_id is None:
        return
    # Some tokenizers (e.g. Qwen) define a bos_token that the chat template never emits.
    uses_bos = bool(getattr(tokenizer, "add_bos_token", True)) or n > 0
    if not uses_bos:
        return
    if n != 1 or ids[0] != tokenizer.bos_token_id:
        raise ValueError(f"Expected exactly one BOS at position 0, found {n} (first id {ids[:1]})")
