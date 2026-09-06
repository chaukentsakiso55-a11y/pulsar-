from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from model.pulsar1 import PulsarConfig, PulsarLM
from model.tokenizer import ByteTokenizer
from pulsar.providers.base import Provider
from pulsar.schemas import Message


class Pulsar1Provider(Provider):
    name = "pulsar-1"

    def __init__(self, checkpoint: str):
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(path, map_location=self.device, weights_only=False)
        cfg = PulsarConfig(**payload["config"])
        self.model = PulsarLM(cfg).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.tokenizer = ByteTokenizer()

    @staticmethod
    def _format_prompt(messages: list[Message]) -> str:
        lines = []
        for m in messages:
            lines.append(f"<{m.role}>\n{m.content}\n")
        lines.append("<assistant>\n")
        return "".join(lines)

    def _generate_sync(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        prompt = self._format_prompt(messages)
        ids = self.tokenizer.encode(prompt, add_bos=True)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
        )
        new_ids = out[0, len(ids) :].tolist()
        return self.tokenizer.decode(new_ids).strip()

    async def generate(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, messages, max_tokens, temperature
        )
