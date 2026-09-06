from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal

from pulsar.db import Database
from pulsar.router import ModelRouter, RouteDecision
from pulsar.schemas import Message

ReasoningEffort = Literal["fast", "standard", "think", "deep", "max"]


@dataclass(slots=True)
class OrchestrationResult:
    text: str
    provider_id: str
    backend_model: str
    effort: str
    passes: int
    retrieved_chunks: int = 0
    route_score: float = 0.0
    trace: list[str] = field(default_factory=list)


class PulsarOrchestrator:
    """Pulsar Max orchestration layer.

    The orchestrator deliberately keeps internal deliberation private. It asks
    providers for concise plans/critiques and only returns the final answer plus
    high-level routing metadata.
    """

    def __init__(self, router: ModelRouter, db: Database, retrieval_limit: int = 4, max_passes: int = 5):
        self.router = router
        self.db = db
        self.retrieval_limit = max(0, retrieval_limit)
        self.max_passes = max(1, max_passes)

    @staticmethod
    def _latest_user(messages: list[Message]) -> str:
        return next((m.content for m in reversed(messages) if m.role == "user"), "")

    def _with_context(self, messages: list[Message], conversation_id: str | None) -> tuple[list[Message], int]:
        augmented = list(messages)
        context_parts: list[str] = []
        latest = self._latest_user(messages)

        if conversation_id:
            history = self.db.get_conversation(conversation_id, limit=12)
            if history:
                rendered = "\n".join(f"{m['role']}: {m['content']}" for m in history)
                context_parts.append("Relevant conversation memory:\n" + rendered)

        if latest and self.retrieval_limit:
            chunks = self.db.search_knowledge(latest, limit=self.retrieval_limit)
            if chunks:
                rendered = "\n\n".join(f"[{c['source']}] {c['content']}" for c in chunks)
                context_parts.append("Retrieved knowledge:\n" + rendered)
            retrieved = len(chunks)
        else:
            retrieved = 0

        if context_parts:
            augmented.insert(
                0,
                Message(
                    role="system",
                    content=(
                        "Pulsar retrieved private context for this request. Use it only when relevant; "
                        "do not claim it is more certain than the user's question warrants.\n\n"
                        + "\n\n".join(context_parts)
                    ),
                ),
            )
        return augmented, retrieved

    async def run(
        self,
        messages: list[Message],
        model: str,
        effort: ReasoningEffort,
        max_tokens: int,
        temperature: float,
        conversation_id: str | None = None,
        verify: bool = True,
    ) -> OrchestrationResult:
        decision = self.router.route(model=model, effort=effort)
        provider = decision.provider
        enriched, retrieved = self._with_context(messages, conversation_id)
        trace = [f"route:{decision.provider_id}"]

        # Fast and standard preserve low latency and cost.
        if effort in {"fast", "standard"}:
            text = await provider.generate(enriched, max_tokens, temperature)
            passes = 1
        elif effort == "think":
            plan = await provider.generate(
                [
                    Message(role="system", content="Create a concise solution plan. Do not provide hidden chain-of-thought; list only key checks and approach."),
                    *enriched,
                ],
                min(512, max_tokens),
                0.2,
            )
            trace.append("plan")
            text = await provider.generate(
                [
                    Message(role="system", content=f"Use this concise plan as guidance, then answer the user directly and accurately:\n{plan}"),
                    *enriched,
                ],
                max_tokens,
                temperature,
            )
            passes = 2
        else:
            # Deep/Max: gather independent expert drafts concurrently, then synthesize.
            expert_count = 2 if effort == "deep" else 3
            expert_roles = [
                "Solve the request carefully. Focus on correctness and completeness.",
                "Independently solve the request. Look for edge cases and mistaken assumptions.",
                "Act as a verifier. Identify facts, calculations, or code that deserve checking, then propose the best answer.",
            ][:expert_count]
            expert_calls = [
                provider.generate([Message(role="system", content=role), *enriched], max_tokens, max(0.1, temperature - 0.15))
                for role in expert_roles
            ]
            drafts = await asyncio.gather(*expert_calls)
            trace.append(f"experts:{expert_count}")
            synthesis_prompt = "\n\n--- CANDIDATE ---\n".join(drafts)
            text = await provider.generate(
                [
                    Message(
                        role="system",
                        content=(
                            "You are Pulsar Max's synthesis stage. Produce one final answer using the strongest parts "
                            "of the candidates below. Resolve conflicts, remove unsupported claims, do not mention "
                            "the candidates or internal process, and answer the original user directly.\n\n"
                            + synthesis_prompt
                        ),
                    ),
                    *enriched,
                ],
                max_tokens,
                temperature,
            )
            passes = expert_count + 1

        if verify and effort in {"deep", "max"} and passes < self.max_passes:
            critique = await provider.generate(
                [
                    Message(
                        role="system",
                        content=(
                            "Check the draft below for factual errors, logical mistakes, contradictions, unsafe instructions, "
                            "or missing caveats. Return only a concise correction memo; say OK if no meaningful correction is needed.\n\n"
                            + text
                        ),
                    )
                ],
                min(700, max_tokens),
                0.1,
            )
            passes += 1
            trace.append("verify")
            if critique.strip().upper() != "OK" and passes < self.max_passes:
                text = await provider.generate(
                    [
                        Message(
                            role="system",
                            content=(
                                "Revise the draft using the correction memo. Return only the improved final answer.\n\n"
                                f"DRAFT:\n{text}\n\nCORRECTION MEMO:\n{critique}"
                            ),
                        )
                    ],
                    max_tokens,
                    max(0.1, temperature - 0.1),
                )
                passes += 1
                trace.append("revise")

        if conversation_id:
            latest = self._latest_user(messages)
            if latest:
                self.db.add_message(conversation_id, "user", latest)
            self.db.add_message(conversation_id, "assistant", text)

        return OrchestrationResult(
            text=text,
            provider_id=decision.provider_id,
            backend_model=decision.backend_model,
            effort=effort,
            passes=passes,
            retrieved_chunks=retrieved,
            route_score=decision.score,
            trace=trace,
        )
