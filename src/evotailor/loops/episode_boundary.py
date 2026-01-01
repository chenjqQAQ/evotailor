from __future__ import annotations

import json
from typing import Any

from evotailor.contracts import TrialResult

SYSTEM_PROMPT = """You are an episode-boundary controller for an experiment. Decide only whether the current episode should continue or end. Respect the hard limits supplied by the kernel. Do not change total budget, search space, objective, or completed results. Return strict JSON: decision (continue or end_episode), reason, confidence, evidence_refs. Keep enum values and IDs in English; write explanations in Simplified Chinese."""


class EpisodeBoundaryAgent:
    def __init__(self, client=None) -> None:
        self.client = client

    def decide(
        self,
        *,
        episode: int,
        episode_trials: int,
        completed_trials: int,
        total_trials: int,
        min_trials: int,
        max_trials: int,
        episode_count: int,
        min_episodes: int,
        max_episodes: int,
        history: list[TrialResult],
        current_episode: list[TrialResult],
        use_llm: bool,
    ) -> dict[str, Any]:
        remaining = total_trials - completed_trials
        base = {
            "decision": "continue",
            "reason": "当前 Episode 尚未达到安全复盘条件。",
            "confidence": 1.0,
            "evidence_refs": [],
            "used_fallback": True,
        }
        if remaining == 0:
            base.update(
                decision="end_episode",
                reason="Campaign 已用尽预算。",
                evidence_refs=[x.trial_id for x in current_episode],
            )
            return base
        if episode_trials < min_trials:
            base.update(
                reason=f"当前 Episode 仅完成 {episode_trials} 个 Trial，低于最小长度 {min_trials}。",
                evidence_refs=[x.trial_id for x in current_episode],
            )
            return base
        if episode_trials >= max_trials or (remaining == 0):
            base.update(
                decision="end_episode",
                reason="达到 Episode 最大长度或 Campaign 已用尽预算。",
                evidence_refs=[x.trial_id for x in current_episode],
            )
            return base
        if episode_count >= max_episodes:
            base.update(reason="已达到最大 Episode 数，继续当前 Episode。")
            return base
        if 0 < remaining < min_trials:
            base.update(reason="剩余预算不能组成完整的下一 Episode，继续当前 Episode。")
            return base
        required_episodes = max(min_episodes - episode_count, 0)
        if (required_episodes and remaining == required_episodes * min_trials) or (
            remaining == min_trials and remaining > max_trials - episode_trials
        ):
            base.update(decision="end_episode", reason="为后续 Episode 保留满足最小长度的预算。")
            return base
        if not use_llm or self.client is None:
            return base
        payload = {
            "episode": episode,
            "episode_trials": episode_trials,
            "completed_trials": completed_trials,
            "remaining_trials": remaining,
            "total_trials": total_trials,
            "episode_count": episode_count,
            "min_trials": min_trials,
            "max_trials": max_trials,
            "min_episodes": min_episodes,
            "max_episodes": max_episodes,
            "current_episode": [x.model_dump(mode="json") for x in current_episode[-12:]],
            "history_summary": [
                {"trial_id": x.trial_id, "objective": x.objective, "status": x.status.value}
                for x in history[-24:]
            ],
        }
        try:
            answer = self.client.complete_json(
                SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False, default=str)
            )
            decision = str(answer.get("decision", "continue"))
            if decision not in {"continue", "end_episode"}:
                raise ValueError("invalid boundary decision")
            answer["decision"] = decision
            answer["reason"] = str(answer.get("reason", "Agent 未提供理由。"))
            # A worded value here ("中等偏低") used to raise and end the Campaign.
            # Confidence is an annotation; the decision field is what matters.
            try:
                number = float(answer.get("confidence", 0.5))
            except (TypeError, ValueError):
                number = 0.5
            answer["confidence"] = max(0.0, min(1.0, number))
            answer["evidence_refs"] = (
                answer.get("evidence_refs") if isinstance(answer.get("evidence_refs"), list) else []
            )
            answer["used_fallback"] = False
            return answer
        except Exception as exc:
            raise RuntimeError("Episode boundary model decision failed; Trial is saved") from exc
