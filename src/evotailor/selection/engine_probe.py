"""Ask the real engine whether it accepts a value, before the space is frozen.

A flag appearing in ``vllm serve --help=all`` does not mean the flag is usable.
``--enable-dbo`` is listed by the installed vLLM and then refused during config
validation, because microbatching requires the DeepEP kernels and a deepep
all2all backend that this platform does not have. A search space containing that
value would spend a large share of its budget on Trials that cannot start.

Added categorical values are checked by starting the served model and issuing a real request.
For numeric dimensions only the domain endpoints are probed; this is recorded as partial coverage.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from evotailor.selection.resources import serving_lease
from evotailor.spaces.search_space import Dimension

#: Container port the probe server listens on. Distinct from the Trial port so a
#: probe can never collide with a running Campaign.
PROBE_PORT = 18099


class EngineValueProbe:
    """Start the real engine with one configuration value and report the outcome."""

    def __init__(
        self,
        runtime: dict[str, Any],
        *,
        tiny_model: Path,
        port: int = PROBE_PORT,
        timeout_s: int = 300,
        max_model_len: int = 2048,
    ) -> None:
        self.runtime = runtime
        self.tiny_model = Path(tiny_model)
        self.port = port
        self.timeout_s = timeout_s
        self.max_model_len = max_model_len
        self.container_name = "evotailor-probe-" + uuid.uuid4().hex

    def _command(self, flags: list[str]) -> list[str]:
        environment_root = str(self.runtime.get("vllm_environment_path") or "")
        image = str(self.runtime.get("image") or "")
        gpu_uuids = [str(item) for item in self.runtime.get("gpu_uuids") or []]
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            self.container_name,
            "--runtime=nvidia",
            "-e",
            f"NVIDIA_VISIBLE_DEVICES={','.join(gpu_uuids)}",
            "-v",
            f"{environment_root}:{environment_root}:ro",
            "-v",
            f"{self.tiny_model}:/models/probe:ro",
            "-p",
            f"{self.port}:{self.port}",
            image,
            f"{environment_root}/bin/vllm",
            "serve",
            "/models/probe",
            "--served-model-name",
            "probe",
            # The port must be reachable from the host, so it cannot bind to the
            # container's own loopback.
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--max-model-len",
            str(self.max_model_len),
            *flags,
        ]

    def probe(self, flags: list[str]) -> tuple[bool, str]:
        """Return whether the engine started, and the evidence either way."""
        if not self.tiny_model.is_dir():
            return False, f"the probe model directory does not exist: {self.tiny_model}"
        with (
            serving_lease(
                self.runtime.get("gpu_uuids") or [], f"http://127.0.0.1:{self.port}/health"
            ),
            tempfile.TemporaryFile(mode="w+") as log,
        ):
            process = subprocess.Popen(
                self._command(flags),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.timeout_s
            started = False
            try:
                while time.monotonic() < deadline:
                    if self._healthy():
                        started = True
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(3)
            finally:
                self._stop(process)
                removal = subprocess.run(
                    ["docker", "rm", "-f", self.container_name],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if removal.returncode and "No such container" not in removal.stderr:
                    raise RuntimeError(f"Probe cleanup failed: {removal.stderr[-300:]}")
            log.seek(0)
            output = log.read()
        if started:
            return True, "the engine started and completed a real inference request"
        return False, self._reason(output)

    def _inference(self) -> bool:
        body = json.dumps(
            {
                "model": "probe",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 8,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        return (
            bool(result.get("choices"))
            and (result.get("usage") or {}).get("completion_tokens", 0) > 0
        )

    def _healthy(self) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    @staticmethod
    def _stop(process: subprocess.Popen[str]) -> None:
        try:
            process.terminate()
            process.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            try:
                process.kill()
                process.wait(timeout=15)
            except (subprocess.TimeoutExpired, OSError):
                pass

    @staticmethod
    def _reason(output: str) -> str:
        """Keep the engine's own complaint rather than paraphrasing it.

        The most useful line is the substantive one, not the pydantic
        documentation URL that follows it, so those are skipped and a line that
        names the actual constraint is preferred.
        """
        candidates = [
            line.strip()
            for line in output.splitlines()
            if any(
                token in line
                for token in (
                    "Error",
                    "error",
                    "Assertion",
                    "unrecognized",
                    "invalid",
                    "not supported",
                )
            )
            and "errors.pydantic.dev" not in line
        ]
        for line in candidates:
            if any(
                token in line
                for token in ("Assertion failed", "not supported", "unrecognized", "invalid")
            ):
                return line[:400]
        if candidates:
            return candidates[-1][:400]
        tail = [line.strip() for line in output.splitlines() if line.strip()]
        return tail[-1][:400] if tail else "the engine exited without reporting a reason"


def flags_for_value(spec: dict[str, Any], value: Any) -> list[str]:
    """Turn one dimension value into the flags the engine would receive."""
    if isinstance(value, bool):
        flag = spec.get("true_flag") if value else spec.get("false_flag")
        return [str(flag)] if flag else []
    flag = spec.get("flag")
    return [str(flag), str(value)] if flag else []


def probe_dimension_values(
    dimension: Dimension, spec: dict[str, Any], probe: EngineValueProbe
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Probe each value of a boolean dimension and keep the ones that start.

    Returns the usable values and the evidence for every probe. A boolean feature
    flag the platform cannot enable is not a tunable dimension, so a dimension
    left with fewer than two usable values is reported as unusable rather than
    silently narrowed.
    """
    values = dimension.values()
    if dimension.kind in ("integer", "float") and values:
        values = list(dict.fromkeys((values[0], values[-1])))
    usable: list[Any] = []
    evidence: list[dict[str, Any]] = []
    for value in values:
        flags = flags_for_value(spec, value)
        ok, reason = probe.probe(flags)
        evidence.append({"value": value, "flags": flags, "accepted": ok, "reason": reason})
        if ok:
            usable.append(value)
    return usable, evidence


def load_runtime(path: Path) -> dict[str, Any]:
    """Read the runtime YAML that describes how the engine is launched."""
    import yaml

    try:
        payload = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def probe_from_environment() -> EngineValueProbe | None:
    """Build a probe from the environment, or None when it is not configured.

    The probe model is a deployment detail, so it is supplied by the environment
    rather than guessed. When it is absent the design step says so and keeps the
    dimension, because silently dropping a justified dimension would be worse
    than the risk it carries.
    """
    import os

    runtime_path = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
    if not runtime_path:
        return None
    runtime = load_runtime(Path(runtime_path))
    if not runtime:
        return None
    # The probe only asks whether the engine ACCEPTS a flag value, so it must not pay
    # for the served model's weights. With a 27B checkpoint the start-up exceeds the
    # probe timeout and the design step never finishes. ``probe_model_path`` names a
    # small checkpoint for this one question; ``model_path`` remains the fallback.
    model = runtime.get("probe_model_path") or runtime.get("model_path")
    if not model:
        return None
    return EngineValueProbe(runtime, tiny_model=Path(model))


def summarise(evidence: list[dict[str, Any]]) -> str:
    return json.dumps(evidence, ensure_ascii=False, default=str)


# Preserve the public helper name for existing callers.
probe_boolean_values = probe_dimension_values
