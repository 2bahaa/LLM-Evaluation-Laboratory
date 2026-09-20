"""Prompt templates. Each template turns an Example into a list of chat messages."""
from __future__ import annotations

from dataclasses import dataclass

from .datasets import Example


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str = "{question}"
    shots: tuple[tuple[str, str], ...] = ()  # (question, answer) demonstrations

    def to_messages(self, example: Example) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system}]
        for question, answer in self.shots:
            messages.append({"role": "user", "content": self.user.format(question=question)})
            messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": self.user.format(question=example.question)})
        return messages


_CONCISE = "Answer with a short phrase only. Do not explain and do not write full sentences."

_TEMPLATES: dict[str, PromptTemplate] = {
    t.name: t
    for t in (
        PromptTemplate(
            name="zero_shot",
            system="You are a helpful assistant.",
        ),
        PromptTemplate(
            name="concise",
            system=_CONCISE,
            user="Question: {question}\nAnswer:",
        ),
        PromptTemplate(
            name="few_shot",
            system=_CONCISE,
            user="Question: {question}\nAnswer:",
            shots=(
                ("What is the capital of Italy?", "Rome"),
                ("How many days are in a week?", "7"),
                ("What is the opposite of hot?", "Cold"),
            ),
        ),
    )
}


def register_template(template: PromptTemplate) -> None:
    """Add (or replace) a template so it can be referenced by name in a config."""
    _TEMPLATES[template.name] = template


def get_template(name: str) -> PromptTemplate:
    try:
        return _TEMPLATES[name]
    except KeyError:
        raise KeyError(f"Unknown prompt template '{name}'. Available: {', '.join(sorted(_TEMPLATES))}") from None


def available_templates() -> list[str]:
    return sorted(_TEMPLATES)
