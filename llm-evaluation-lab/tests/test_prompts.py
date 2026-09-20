import pytest

from llm_eval_lab.datasets import Example
from llm_eval_lab.prompts import PromptTemplate, available_templates, get_template, register_template

EX = Example(id="1", question="What is 2+2?", answers=["4"])


def test_builtin_templates_exist():
    assert {"zero_shot", "concise", "few_shot"} <= set(available_templates())


def test_zero_shot_messages():
    messages = get_template("zero_shot").to_messages(EX)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[-1]["content"] == "What is 2+2?"


def test_few_shot_interleaves_demonstrations_before_the_question():
    messages = get_template("few_shot").to_messages(EX)
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert messages[-1]["content"] == "Question: What is 2+2?\nAnswer:"
    assert messages[2]["content"] == "Rome"


def test_unknown_template_lists_alternatives():
    with pytest.raises(KeyError, match="zero_shot"):
        get_template("nope")


def test_register_custom_template():
    register_template(PromptTemplate(name="shout", system="Answer in capitals.", user="{question}!"))
    assert get_template("shout").to_messages(EX)[-1]["content"] == "What is 2+2?!"
