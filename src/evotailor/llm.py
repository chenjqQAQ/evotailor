from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class _Endpoint:
    """One OpenAI-compatible relay the client may talk to."""

    label: str
    base_url: str
    api_key_env: str

    def key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
        return value


class OpenAICompatibleClient:
    """Chat client with a permanent failover relay.

    The primary relay rate-limits under load, and a rate limit in the middle of a
    Campaign previously raised straight out of the selector and ended the run. The
    relay that backs this experiment has a documented stand-in serving the same
    model, so a primary that cannot answer is abandoned for the rest of the run
    rather than retried until the Campaign dies. The switch is one-way and recorded,
    because silently alternating relays would make the audit trail ambiguous.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "EVOTAILOR_PROPOSER_API_KEY",
        timeout_s: float = 60,
        retries: int = 2,
        *,
        fallback_base_url: str | None = None,
        fallback_api_key_env: str | None = None,
        max_backoff_s: float = 300.0,
        requests_per_endpoint: int | None = None,
    ) -> None:
        self.primary = _Endpoint("primary", base_url.rstrip("/"), api_key_env)
        self.fallback = (
            _Endpoint("fallback", fallback_base_url.rstrip("/"), fallback_api_key_env)
            if fallback_base_url and fallback_api_key_env
            else None
        )
        self.model = model
        self.timeout_s = timeout_s
        #: Attempts per endpoint. Prefer an explicit value, else the configured retries.
        self.retries = max(int(retries), 0)
        self.requests_per_endpoint = requests_per_endpoint or max(self.retries + 1, 1)
        self.max_backoff_s = max_backoff_s
        self._active = self.primary
        self._switched_at: str | None = None
        self._switch_reason: str | None = None

    # -- relay selection ---------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._active.base_url

    @property
    def api_key_env(self) -> str:
        return self._active.api_key_env

    @property
    def active_endpoint(self) -> str:
        return self._active.label

    def failover_record(self) -> dict[str, Any]:
        """What the audit needs to know about relay usage."""
        return {
            "active_endpoint": self._active.label,
            "switched_at": self._switched_at,
            "switch_reason": self._switch_reason,
            "fallback_configured": self.fallback is not None,
        }

    def _switch_to_fallback(self, reason: str) -> None:
        if self.fallback is None or self._active is self.fallback:
            return
        self._active = self.fallback
        self._switched_at = datetime.now(UTC).isoformat()
        self._switch_reason = reason[:400]
        self._audit(
            {
                "event": "relay_failover",
                "timestamp": self._switched_at,
                "from": self.primary.label,
                "to": self.fallback.label,
                "reason": self._switch_reason,
            }
        )

    # -- connectivity ------------------------------------------------------

    def check_connectivity(self) -> dict[str, Any]:
        """Verify the configured relay before any Trial budget is consumed."""
        key = self._active.key()
        try:
            with httpx.Client(timeout=min(self.timeout_s, 20), trust_env=False) as client:
                response = client.get(
                    f"{self._active.base_url}/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise RuntimeError(
                "model relay connectivity check failed: "
                f"{type(exc).__name__}: {exc}; base_url={self._active.base_url}"
            ) from exc
        model_ids = {
            str(item.get("id"))
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        }
        return {
            "base_url": self._active.base_url,
            "endpoint": self._active.label,
            "model": self.model,
            "listed": self.model in model_ids,
        }

    # -- requests ----------------------------------------------------------

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            return self._complete_on(self._active, system, user)
        except Exception as exc:
            if self.fallback is None or self._active is self.fallback:
                raise
            # One-way: the primary has now failed for this whole Campaign.
            self._switch_to_fallback(f"{type(exc).__name__}: {exc}")
            return self._complete_on(self._active, system, user)

    def _complete_on(self, endpoint: _Endpoint, system: str, user: str) -> dict[str, Any]:
        key = endpoint.key()
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        # The configured GPT-6 relay rejects temperature=0 as unsupported_value.
        # Omit this optional field instead of retrying the same invalid request.
        if not self.model.startswith("gpt-6"):
            payload["temperature"] = 0
        last: Exception | None = None
        for attempt in range(self.requests_per_endpoint):
            try:
                with httpx.Client(timeout=self.timeout_s, trust_env=False) as client:
                    response = client.post(
                        f"{endpoint.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json=payload,
                    )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                answer = self._parse_json(content)
                self._audit(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "endpoint": endpoint.label,
                        "base_url": endpoint.base_url,
                        "requested_model": self.model,
                        "response_model": result.get("model"),
                        "response_id": result.get("id"),
                        "usage": result.get("usage"),
                        "attempt": attempt + 1,
                        "prompt_sha256": hashlib.sha256((system + user).encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
                        "decision": answer,
                    }
                )
                return answer
            except Exception as exc:
                last = exc
                if attempt + 1 < self.requests_per_endpoint:
                    time.sleep(self._backoff(exc, attempt))
        raise RuntimeError(
            f"model request failed after {self.requests_per_endpoint} attempts on "
            f"{endpoint.label}: {type(last).__name__}: {last}"
        ) from last

    def _backoff(self, exc: Exception, attempt: int) -> float:
        """Wait longer for a rate limit than for a transient transport error.

        A rate limit without a Retry-After header keeps the long-standing 60s-per-attempt
        cooldown: retrying a saturated relay eagerly just hits the same limit again. The
        only new behaviour is the cap, so a relay that asks for hours is handed to the
        fallback instead of stalling the Campaign.
        """
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                parsed = self._retry_after_seconds(retry_after)
                if parsed is not None:
                    return min(max(parsed, 0.0), self.max_backoff_s)
            return min(60.0 * (attempt + 1), self.max_backoff_s)
        return min(2.0 ** attempt, 4.0)

    @staticmethod
    def _retry_after_seconds(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            pass
        from email.utils import parsedate_to_datetime

        try:
            deadline = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        return max((deadline - datetime.now(UTC)).total_seconds(), 0.0)

    @staticmethod
    def _audit(entry: dict[str, Any]) -> None:
        destination = os.environ.get("EVOTAILOR_LLM_AUDIT_PATH")
        if not destination:
            return
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.S)
            if not match:
                raise
            return json.loads(match.group(0))


def client_from_config(config: Any) -> OpenAICompatibleClient:
    """Build the Campaign's client, including any configured stand-in relay."""
    return OpenAICompatibleClient(
        config.base_url,
        config.model,
        config.api_key_env,
        config.model_timeout_s,
        config.model_retries,
        fallback_base_url=getattr(config, "fallback_base_url", None),
        fallback_api_key_env=getattr(config, "fallback_api_key_env", None),
        max_backoff_s=float(getattr(config, "model_max_backoff_s", 300.0)),
        requests_per_endpoint=int(getattr(config, "model_attempts_per_endpoint", 3)),
    )
