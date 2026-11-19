"""Context policies (PREREG section 2). The message layout is pinned by the contract tests."""
from __future__ import annotations


def build_messages(policy: str, system: str, question: str,
                   history: list[tuple[str, str]]) -> list[dict]:
    """Build the request messages for one step.

    history is a list of (assistant_text, observation_text) pairs from earlier steps.
    User message layout: "Question: {q}" and "Observation: {obs}".
    """
    name, _, arg = policy.partition(":")
    if name in ("carry_all", "reset_summary_broken"):
        pairs = list(history)  # broken control writes notes but drops nothing
    elif name == "reset_summary":
        pairs = history[-1:]  # only the latest action/observation pair is carried
    elif name == "sliding_window":
        n = int(arg)
        pairs = history[-n:] if n > 0 else []
    else:
        raise ValueError(f"unknown context policy: {policy!r}")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}"},
    ]
    for action_text, observation in pairs:
        messages.append({"role": "assistant", "content": action_text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    return messages
