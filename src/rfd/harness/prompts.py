"""Prompt builders.

Marked stepwise CoT: "Step k: ... <a> <op> <b> = <c>" lines, then
"Final Answer: <number>". Two worked demonstrations are taken from the GSM8K
*train* split (Natalia: train[0]; Betty: train[2]); GSM-Symbolic templates are
built from GSM8K *test*, so the demos are disjoint from every evaluated problem.
"""

from __future__ import annotations

from typing import Any

COT_SYSTEM = (
    "You are a careful math solver. Solve the problem step by step. Write each step on "
    "its own line, starting with 'Step k:'. Each step does exactly one arithmetic "
    "operation (+, -, *, /), written as '<number> <op> <number> = <number>' with plain "
    "numbers (no units or LaTeX inside the calculation). After the last step, write "
    "'Final Answer: <number>'."
)

COT_DEMOS = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?",
        "Step 1: In May she sold half as many clips as in April: 48 / 2 = 24.\n"
        "Step 2: Altogether she sold 48 + 24 = 72.\n"
        "Final Answer: 72",
    ),
    (
        "Betty is saving money for a new wallet which costs $100. Betty has only half of the "
        "money she needs. Her parents decided to give her $15 for that purpose, and her "
        "grandparents twice as much as her parents. How much more money does Betty need to "
        "buy the wallet?",
        "Step 1: Betty has half of the money: 100 / 2 = 50.\n"
        "Step 2: Her grandparents give twice as much as her parents: 15 * 2 = 30.\n"
        "Step 3: Her family gives her 15 + 30 = 45.\n"
        "Step 4: Together she has 50 + 45 = 95.\n"
        "Step 5: She still needs 100 - 95 = 5.\n"
        "Final Answer: 5",
    ),
]


COT_REMINDER = (
    "(Use the same format: one calculation per step, plain numbers, no LaTeX, then "
    "'Final Answer: <number>'.)"
)


def cot_messages(question: str) -> list[dict[str, str]]:
    """System instruction, two demo turns, then the problem plus a format
    reminder (the reminder raised Final-Answer-marker adherence; DECISIONS D9)."""
    msgs = [{"role": "system", "content": COT_SYSTEM}]
    for q, a in COT_DEMOS:
        msgs += [{"role": "user", "content": f"Problem: {q}"}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Problem: {question}\n\n{COT_REMINDER}"})
    return msgs


# Sun et al. (2507.12379, App. A) arithmetic format, adapted to +/-.
ARITH_SYSTEM = {
    "+": (
        "You are a helpful assistant that calculates the sum of two numbers. Always provide "
        "your answer in the format <<x+y=z>>. Do not provide any additional explanation."
    ),
    "-": (
        "You are a helpful assistant that calculates the difference of two numbers. Always "
        "provide your answer in the format <<x-y=z>>. Do not provide any additional "
        "explanation."
    ),
}
_ARITH_VERB = {"+": "sum", "-": "difference"}


def arith_messages(a: int, b: int, op: str, demos: list[tuple[int, int]]) -> list[dict[str, str]]:
    def user(x: int, y: int) -> str:
        return (
            f"Calculate the {_ARITH_VERB[op]} of the following two numbers:\n"
            f"First number: {x}\nSecond number: {y}"
        )

    msgs = [{"role": "system", "content": ARITH_SYSTEM[op]}]
    for x, y in demos:
        z = x + y if op == "+" else x - y
        msgs += [
            {"role": "user", "content": user(x, y)},
            {"role": "assistant", "content": f"<<{x}{op}{y}={z}>>"},
        ]
    msgs.append({"role": "user", "content": user(a, b)})
    return msgs


def merge_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages or messages[0]["role"] != "system":
        return messages
    sys_text, rest = messages[0]["content"], [dict(m) for m in messages[1:]]
    rest[0]["content"] = f"{sys_text}\n\n{rest[0]['content']}"
    return rest


def chat_ids(
    tokenizer: Any, messages: list[dict[str, str]], assistant_prefix: str = ""
) -> list[int]:
    """Token ids of the chat prompt (generation prompt added), plus an optional
    teacher-forced start of the assistant turn. Templates without a system role
    (Gemma-2) get the system text prepended to the first user turn."""
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 - jinja raises on unsupported roles
        text = tokenizer.apply_chat_template(
            merge_system(messages), tokenize=False, add_generation_prompt=True
        )
    return tokenizer(text + assistant_prefix, add_special_tokens=False)["input_ids"]
