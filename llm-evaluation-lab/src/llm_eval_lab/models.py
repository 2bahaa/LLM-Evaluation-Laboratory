"""Model backends. Anything with ``name``, ``generate(messages)`` and ``close()`` can be benchmarked."""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Protocol

from .config import GenerationSpec, ModelSpec
from .utils import select_device


@dataclass
class Generation:
    text: str
    latency_s: float  # wall-clock time of the generate() call only
    new_tokens: int


class LLM(Protocol):
    name: str

    def generate(self, messages: list[dict[str, str]]) -> Generation: ...

    def close(self) -> None: ...


class HFCausalLM:
    """A local Hugging Face causal language model running on PyTorch."""

    def __init__(self, name: str, hf_id: str, generation: GenerationSpec, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.generation = generation
        self.device = select_device(device)

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(hf_id)
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Base models without a chat template: fall back to a plain transcript.
        lines = []
        for m in messages:
            lines.append(m["content"] if m["role"] == "system" else f"{m['role'].capitalize()}: {m['content']}")
        return "\n".join(lines) + "\nAssistant:"

    def _sync(self) -> None:
        if self.device == "cuda":
            import torch

            torch.cuda.synchronize()

    def generate(self, messages: list[dict[str, str]]) -> Generation:
        import torch

        prompt = self._render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        kwargs: dict = {
            "max_new_tokens": self.generation.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.generation.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.generation.temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)

        self._sync()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        self._sync()
        latency = time.perf_counter() - start

        new_ids = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return Generation(text=text, latency_s=latency, new_tokens=int(new_ids.shape[0]))

    def close(self) -> None:
        """Free memory so the next model in the experiment starts from a clean slate."""
        import torch

        del self.model
        del self.tokenizer
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()


def load_model(spec: ModelSpec, generation: GenerationSpec, device: str | None = None) -> HFCausalLM:
    return HFCausalLM(spec.name, spec.hf_id, generation, device)
