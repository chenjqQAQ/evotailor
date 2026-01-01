"""Workload profile extracted from the upper task's own code and configuration.

Two questions drive this module:

* **What is the task?** The space designer needs the real entry point, the
  inference stack, and the served model. A list of file names answers none of
  those, which is why the earlier discovery could only ever decline to add a
  dimension.
* **What shape of traffic does it produce?** Concurrency, context length,
  arrival pattern, and per-step call depth are the things that justify adding a
  tuning dimension. A dimension that cannot be tied back to one of them is a
  guess, and guessing is exactly what this project must not do.

Everything here is derived from code and configuration, so a first-ever
Campaign with no run history still gets a usable profile. Every value is
wrapped in a *fact record* naming where it came from and how strong that
evidence is; a value that could not be established is recorded as
``unavailable`` rather than invented, and a value produced by a stated rule
rather than a measurement is recorded as ``inferred``.

Fact records are addressable by a dotted id (for example
``workload.scenario.agent_count``). The space designer must cite one of those
ids for every dimension it adds, so the design step can be audited instead of
trusted.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: A value read from a command's real output or a real tokenizer run.
GRADE_MEASURED = "measured"
#: A value read verbatim out of a source or configuration artifact.
GRADE_DECLARED = "declared"
#: A value produced by a stated rule over declared values, not observed directly.
GRADE_INFERRED = "inferred"
#: A value that could not be established from the available evidence.
GRADE_UNAVAILABLE = "unavailable"

_GRADES = frozenset({GRADE_MEASURED, GRADE_DECLARED, GRADE_INFERRED, GRADE_UNAVAILABLE})

#: Keys that identify a fact record. ``is_fact`` requires all three.
_FACT_KEYS = ("value", "source", "grade")

_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        ".deps",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "__pycache__",
        "runs",
    }
)

#: Libraries that show how the task reaches its model endpoint.
_CLIENT_LIBRARIES = (
    "litellm",
    "openai",
    "anthropic",
    "transformers",
    "vllm",
    "ray",
    "httpx",
    "aiohttp",
    "requests",
)

_MAX_SCANNED_FILES = 4000


def fact(value: Any, *, grade: str, source: str, note: str | None = None) -> dict[str, Any]:
    """Build a fact record.

    ``source`` must name the artifact the value came from, so a reader can go
    and check it. ``note`` carries the rule for an inferred value.
    """
    if grade not in _GRADES:
        raise ValueError(f"unknown evidence grade: {grade!r}")
    record: dict[str, Any] = {"value": value, "source": source, "grade": grade}
    if note:
        record["note"] = note
    return record


def unavailable(source: str, note: str) -> dict[str, Any]:
    """Record that a fact could not be established, and why."""
    return fact(None, grade=GRADE_UNAVAILABLE, source=source, note=note)


def is_fact(node: Any) -> bool:
    """True when ``node`` is a fact record rather than a plain container."""
    return (
        isinstance(node, dict)
        and all(key in node for key in _FACT_KEYS)
        and node.get("grade") in _GRADES
    )


def flatten_facts(payload: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Map every fact in ``payload`` to its dotted id.

    This is the addressable view of a profile: the design agent cites these ids,
    and the gate refuses a dimension whose citation is not in this mapping.
    """
    found: dict[str, dict[str, Any]] = {}
    if is_fact(payload):
        if prefix:
            found[prefix] = payload
        return found
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            found.update(flatten_facts(value, child))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.update(flatten_facts(value, f"{prefix}[{index}]"))
    return found


def _walk(root: Path, suffixes: Iterable[str] | None = None) -> list[Path]:
    """Bounded recursive listing that never descends into ignored directories."""
    wanted = {item.lower() for item in suffixes} if suffixes else None
    found: list[Path] = []
    for current, directories, names in os.walk(root):
        directories[:] = sorted(name for name in directories if name not in _IGNORED_DIRECTORIES)
        for name in sorted(names):
            if wanted is not None and Path(name).suffix.lower() not in wanted:
                continue
            found.append(Path(current) / name)
            if len(found) >= _MAX_SCANNED_FILES:
                return found
    return found


def _read_text(path: Path, limit: int = 400_000) -> str | None:
    try:
        return path.read_text(errors="ignore")[:limit]
    except OSError:
        return None


def _load_toml(path: Path) -> dict[str, Any] | None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        return None
    try:
        return tomllib.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError:  # pragma: no cover - optional dependency
        return None
    try:
        return yaml.safe_load(path.read_text())
    except (OSError, ValueError):
        return None


class WorkloadProfiler:
    """Build a workload profile for one upper task.

    ``model_path`` is the model that will actually be served, so the context
    budget is measured against the real tokenizer and the real context limit
    rather than a guess. When it is missing, or its tokenizer cannot be loaded
    offline, the affected facts degrade to ``inferred``/``unavailable`` instead
    of failing: a profile that is thin is still usable, a profile that is wrong
    is not.
    """

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        project_root: Path | None = None,
        scenario_path: Path | None = None,
        history_paths: Iterable[Path] = (),
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.project_root = Path(project_root) if project_root else None
        self.scenario_path = Path(scenario_path) if scenario_path else None
        self.history_paths = [Path(item) for item in history_paths]

    # ------------------------------------------------------------------ entry

    def profile(self, task_root: Path) -> dict[str, Any]:
        """Return ``{"workload": {...}}`` so fact ids do not depend on the caller.

        Mounting the profile under a fixed key keeps the ids the design agent
        cites identical whether it reads the discovery file or the in-memory
        payload.
        """
        root = Path(task_root).resolve()
        scenario = self._find_scenario(root)
        scenario_root = scenario.get("root") or root
        project = (
            Path(self.project_root).resolve()
            if self.project_root
            else self._detect_project_root(scenario_root)
        )
        return {
            "workload": {
                "task": self._task_facts(root, project, scenario),
                "scenario": self._scenario_facts(scenario),
                "context": self._context_facts(scenario),
                "arrival": self._arrival_facts(scenario),
                "history": self._history_facts(),
            }
        }

    @staticmethod
    def _detect_project_root(start: Path) -> Path:
        """Walk up from the scenario to the project that declares the launch.

        The task root is often the scenario directory, which declares nothing
        about how the task starts; the entry points and dependencies live in the
        enclosing project. Walking up finds them without being told.
        """
        for candidate in [start, *start.parents][:8]:
            if (candidate / "pyproject.toml").is_file() or (candidate / "setup.py").is_file():
                return candidate
        return start

    # ------------------------------------------------------------- extraction

    def _find_scenario(self, root: Path) -> dict[str, Any]:
        """Locate the scenario's own declarative artifacts, if it has any.

        A task root may be a whole project holding several scenarios. When more
        than one is found the profile reports the ambiguity instead of profiling
        whichever directory the walk reached first, because profiling the wrong
        scenario would silently misstate the workload.
        """
        if self.scenario_path:
            return self._scenario_at(self.scenario_path.resolve())
        direct = self._scenario_at(root)
        if direct.get("init_config") or direct.get("steps"):
            return direct
        found: set[Path] = set()
        for path in _walk(root, {".json", ".yaml", ".yml"})[:3000]:
            if path.parent.name == "init" and path.name in {"init_config.json", "steps.yaml"}:
                found.add(path.parent.parent)
        if len(found) == 1:
            return self._scenario_at(next(iter(found)))
        if len(found) > 1:
            return {"candidates": sorted(str(item) for item in found)}
        return {}

    @staticmethod
    def _scenario_at(directory: Path) -> dict[str, Any]:
        if directory.name == "init":
            directory = directory.parent
        found: dict[str, Any] = {"root": directory}
        for key, name in (("init_config", "init_config.json"), ("steps", "steps.yaml")):
            candidate = directory / "init" / name
            if candidate.is_file():
                found[key] = candidate
        return found

    def _task_facts(self, root: Path, project: Path, scenario: dict[str, Any]) -> dict[str, Any]:
        entrypoints = self._entrypoints(project)
        return {
            "root": fact(str(root), grade=GRADE_DECLARED, source="task_root"),
            "project_root": fact(
                str(project),
                grade=GRADE_DECLARED,
                source="task_root or the nearest enclosing pyproject.toml/setup.py",
            ),
            "project_name": self._project_name(project),
            "entrypoints": fact(
                entrypoints,
                grade=GRADE_DECLARED,
                source="pyproject.toml / setup.py / Dockerfile / compose / __main__ scan",
                note=(
                    "Read from declarative launch metadata, not guessed from file names."
                    if entrypoints
                    else "No declarative launch metadata and no __main__ guard was found."
                ),
            ),
            "inference_stack": self._inference_stack(project),
            "scenario_artifacts": self._scenario_artifacts_fact(project, scenario),
        }

    @staticmethod
    def _scenario_artifacts_fact(project: Path, scenario: dict[str, Any]) -> dict[str, Any]:
        artifacts = {
            key: str(value) for key, value in scenario.items() if key in {"init_config", "steps"}
        }
        if not artifacts:
            return unavailable(
                "scenario scan", "No init/init_config.json or init/steps.yaml was found."
            )
        rendered = {}
        for key, value in artifacts.items():
            try:
                rendered[key] = str(Path(value).relative_to(project))
            except ValueError:
                rendered[key] = value
        return fact(rendered, grade=GRADE_DECLARED, source="scenario scan")

    def _project_name(self, root: Path) -> dict[str, Any]:
        for name in ("pyproject.toml",):
            path = root / name
            if path.is_file():
                data = _load_toml(path)
                project = (data or {}).get("project") or {}
                if project.get("name"):
                    return fact(
                        str(project["name"]), grade=GRADE_DECLARED, source=f"{name}:[project].name"
                    )
        return unavailable("pyproject.toml", "no [project].name")

    def _entrypoints(self, root: Path) -> list[dict[str, str]]:
        """Real launch metadata only: declared scripts, container commands, guards."""
        found: list[dict[str, str]] = []

        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            data = _load_toml(pyproject)
            scripts = ((data or {}).get("project") or {}).get("scripts") or {}
            for name, target in sorted(scripts.items()):
                found.append(
                    {
                        "value": f"{name} = {target}",
                        "kind": "console_script",
                        "source": "pyproject.toml:[project.scripts]",
                    }
                )

        setup = root / "setup.py"
        if setup.is_file():
            found.extend(self._setup_entry_points(setup))

        for dockerfile in sorted(root.glob("Dockerfile*")):
            text = _read_text(dockerfile)
            if not text:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                match = re.match(r"\s*(CMD|ENTRYPOINT)\s+(.*)", line, re.IGNORECASE)
                if match:
                    found.append(
                        {
                            "value": match.group(2).strip(),
                            "kind": match.group(1).upper(),
                            "source": f"{dockerfile.name}:{line_number}",
                        }
                    )

        for pattern in ("docker-compose*.y*ml", "compose*.y*ml"):
            for compose in sorted(root.glob(pattern)):
                data = _load_yaml(compose)
                services = (data or {}).get("services") if isinstance(data, dict) else None
                if not isinstance(services, dict):
                    continue
                for service, spec in sorted(services.items()):
                    if not isinstance(spec, dict):
                        continue
                    for key in ("command", "entrypoint"):
                        if spec.get(key):
                            found.append(
                                {
                                    "value": json.dumps(spec[key], default=str),
                                    "kind": f"compose.{service}.{key}",
                                    "source": compose.name,
                                }
                            )

        found.extend(self._main_guards(root))
        return found

    @staticmethod
    def _setup_entry_points(setup: Path) -> list[dict[str, str]]:
        """Read ``entry_points`` without executing the setup script."""
        text = _read_text(setup)
        if not text:
            return []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "entry_points" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                    if not isinstance(key, ast.Constant) or "console_scripts" not in str(key.value):
                        continue
                    if not isinstance(value, ast.List):
                        continue
                    return [
                        {
                            "value": str(item.value),
                            "kind": "console_script",
                            "source": "setup.py:entry_points",
                        }
                        for item in value.elts
                        if isinstance(item, ast.Constant)
                    ]
        return []

    @staticmethod
    def _main_guards(root: Path) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        for source in _walk(root, {".py"})[:1500]:
            text = _read_text(source, limit=60_000)
            if not text or "__main__" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.If):
                    continue
                compare = node.test
                if not isinstance(compare, ast.Compare) or not isinstance(compare.left, ast.Name):
                    continue
                if compare.left.id != "__name__":
                    continue
                found.append(
                    {
                        "value": str(source.relative_to(root)),
                        "kind": "__main__ guard",
                        "source": f"{source.relative_to(root)}:{node.lineno}",
                    }
                )
        return found[:40]

    def _inference_stack(self, root: Path) -> dict[str, Any]:
        declared: set[str] = set()
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            data = _load_toml(pyproject)
            project = (data or {}).get("project") or {}
            requirements = list(project.get("dependencies") or [])
            for extra in (project.get("optional-dependencies") or {}).values():
                requirements.extend(extra or [])
            for requirement in requirements:
                name = re.split(r"[<>=!\[;\s]", str(requirement), maxsplit=1)[0].strip().lower()
                if name in _CLIENT_LIBRARIES:
                    declared.add(name)

        imported: set[str] = set()
        for source in _walk(root, {".py"})[:800]:
            text = _read_text(source, limit=40_000)
            if not text:
                continue
            for module in _CLIENT_LIBRARIES:
                if re.search(rf"^\s*(?:from|import)\s+{re.escape(module)}\b", text, re.MULTILINE):
                    imported.add(module)

        servers = self._server_invocations(root)
        libraries = sorted(declared | imported)
        return {
            "client_libraries": fact(
                libraries,
                grade=GRADE_DECLARED if libraries else GRADE_UNAVAILABLE,
                source="pyproject.toml dependencies + import scan",
                note=(
                    None if libraries else "No known model-client library was declared or imported."
                ),
            ),
            "server_invocations": fact(
                servers,
                grade=GRADE_DECLARED if servers else GRADE_UNAVAILABLE,
                source="scan for vllm/sglang/trtllm serve invocations",
                note=None if servers else "No model-server invocation was found in the task tree.",
            ),
        }

    @staticmethod
    def _server_invocations(root: Path) -> list[dict[str, str]]:
        pattern = re.compile(r"\b(vllm|sglang|trtllm)\b[^\n]{0,120}\bserve\b")
        found: list[dict[str, str]] = []
        for source in _walk(root, {".py", ".sh", ".yaml", ".yml", ".toml"})[:1200]:
            text = _read_text(source, limit=80_000)
            if not text:
                continue
            for line_number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    found.append(
                        {
                            "value": line.strip()[:200],
                            "source": f"{source.relative_to(root)}:{line_number}",
                        }
                    )
                    break
            if len(found) >= 10:
                break
        return found

    def _scenario_facts(self, scenario: dict[str, Any]) -> dict[str, Any]:
        init_config = scenario.get("init_config")
        steps_path = scenario.get("steps")

        agents = self._agents(init_config)
        steps = self._steps(steps_path)
        return {
            "ambiguous_scenarios": fact(
                list(scenario.get("candidates") or []),
                grade=GRADE_DECLARED if scenario.get("candidates") else GRADE_UNAVAILABLE,
                source="scenario scan",
                note=(
                    "More than one scenario was found under the task root, so no scenario "
                    "was profiled. Supply the scenario path explicitly."
                    if scenario.get("candidates")
                    else "Exactly one scenario (or none) was found; no ambiguity to report."
                ),
            ),
            "kind": fact(
                "agentsociety" if init_config or steps_path else None,
                grade=GRADE_DECLARED if init_config or steps_path else GRADE_UNAVAILABLE,
                source="init/init_config.json + init/steps.yaml",
                note=None if init_config or steps_path else "Not an AgentSociety-style scenario.",
            ),
            "agent_count": fact(
                len(agents) if agents else None,
                grade=GRADE_DECLARED if agents else GRADE_UNAVAILABLE,
                source=str(init_config) if init_config else "init_config.json",
                note=None if agents else "Agent list could not be read.",
            ),
            "agent_types": fact(
                sorted({item.get("agent_type", "") for item in agents} - {""}) if agents else None,
                grade=GRADE_DECLARED if agents else GRADE_UNAVAILABLE,
                source=str(init_config) if init_config else "init_config.json",
            ),
            "max_react_turns": fact(
                max(
                    [int((item.get("kwargs") or {}).get("max_react_turns") or 0) for item in agents]
                    or [0]
                )
                or None,
                grade=GRADE_DECLARED if agents else GRADE_UNAVAILABLE,
                source="init_config.json:agents[].kwargs.max_react_turns",
                note=("Upper bound on sequential model calls one agent may issue within a step."),
            ),
            "env_modules": fact(
                [
                    str((item or {}).get("module_type"))
                    for item in (self._load_json(init_config) or {}).get("env_modules") or []
                ]
                if init_config
                else None,
                grade=GRADE_DECLARED if init_config else GRADE_UNAVAILABLE,
                source="init_config.json:env_modules",
            ),
            "run_steps": fact(
                steps.get("run_steps"),
                grade=GRADE_DECLARED if steps else GRADE_UNAVAILABLE,
                source="init/steps.yaml",
            ),
            "questionnaire_steps": fact(
                steps.get("questionnaire_steps"),
                grade=GRADE_DECLARED if steps else GRADE_UNAVAILABLE,
                source="init/steps.yaml",
            ),
            "total_steps": fact(
                steps.get("total_steps"),
                grade=GRADE_DECLARED if steps else GRADE_UNAVAILABLE,
                source="init/steps.yaml",
            ),
            "tick_seconds": fact(
                steps.get("tick_seconds"),
                grade=GRADE_DECLARED if steps else GRADE_UNAVAILABLE,
                source="init/steps.yaml:tick",
            ),
            "questions_per_questionnaire": fact(
                steps.get("questions_per_questionnaire"),
                grade=GRADE_DECLARED if steps else GRADE_UNAVAILABLE,
                source="init/steps.yaml:steps[].questions",
            ),
            "scenario_text": self._scenario_text(init_config, steps_path),
        }

    @staticmethod
    def _load_json(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _agents(self, init_config: Path | None) -> list[dict[str, Any]]:
        data = self._load_json(init_config)
        if not data:
            return []
        agents = data.get("agents")
        return [item for item in agents or [] if isinstance(item, dict)]

    def _steps(self, steps_path: Path | None) -> dict[str, Any]:
        data = _load_yaml(steps_path) if steps_path else None
        if not isinstance(data, dict):
            return {}
        entries = [item for item in (data.get("steps") or []) if isinstance(item, dict)]
        run_steps = [item for item in entries if str(item.get("type")) == "run"]
        questionnaires = [item for item in entries if str(item.get("type")) == "questionnaire"]
        ticks = [
            item.get("tick") for item in run_steps if isinstance(item.get("tick"), (int, float))
        ]
        counts = [len(item.get("questions") or []) for item in questionnaires]
        return {
            "run_steps": sum(int(item.get("num_steps") or 0) for item in run_steps),
            "questionnaire_steps": len(questionnaires),
            "total_steps": len(entries),
            "tick_seconds": ticks[0] if ticks else None,
            "questions_per_questionnaire": counts[0] if counts else None,
        }

    def _scenario_text(self, init_config: Path | None, steps_path: Path | None) -> dict[str, Any]:
        """The scenario's own prompt-bearing text, measured with a real tokenizer."""
        chunks: list[str] = []
        sources: list[str] = []
        for agent in self._agents(init_config):
            kwargs = agent.get("kwargs") or {}
            for key in ("bio", "personality"):
                if kwargs.get(key):
                    chunks.append(str(kwargs[key]))
            if chunks:
                sources.append("init_config.json:agents[].kwargs.bio/personality")
                break

        data = _load_yaml(steps_path) if steps_path else None
        if isinstance(data, dict):
            for entry in data.get("steps") or []:
                if not isinstance(entry, dict):
                    continue
                for question in entry.get("questions") or []:
                    if isinstance(question, dict) and question.get("prompt"):
                        chunks.append(str(question["prompt"]))
            if chunks:
                sources.append("init/steps.yaml:steps[].questions[].prompt")

        if not chunks:
            return {
                "sample_count": unavailable("scenario text", "no prompt-bearing text found"),
                "tokens_min": unavailable("scenario text", "no prompt-bearing text found"),
                "tokens_max": unavailable("scenario text", "no prompt-bearing text found"),
            }

        counts = self._token_counts(chunks)
        if counts["grade"] == GRADE_UNAVAILABLE:
            return {
                "sample_count": fact(len(chunks), grade=GRADE_DECLARED, source=" + ".join(sources)),
                "tokens_min": counts,
                "tokens_max": counts,
            }
        token_values = list(counts["value"])
        return {
            "sample_count": fact(len(chunks), grade=GRADE_DECLARED, source=" + ".join(sources)),
            "tokens_min": fact(
                min(token_values),
                grade=counts["grade"],
                source=counts["source"],
                note=counts.get("note"),
            ),
            "tokens_max": fact(
                max(token_values),
                grade=counts["grade"],
                source=counts["source"],
                note=counts.get("note"),
            ),
        }

    def _context_facts(self, scenario: dict[str, Any]) -> dict[str, Any]:
        model = self._model_limits()
        text = self._scenario_text(scenario.get("init_config"), scenario.get("steps"))
        return {
            "model_context_limit": model["context_limit"],
            "model_architecture": model["architecture"],
            "prompt_tokens_min": text["tokens_min"],
            "prompt_tokens_max": text["tokens_max"],
            "context_growth": self._context_growth(scenario),
        }

    def _model_limits(self) -> dict[str, Any]:
        if self.model_path is None:
            reason = "no served model path was supplied to the profiler"
            return {
                "context_limit": unavailable("served model", reason),
                "architecture": unavailable("served model", reason),
            }
        config_path = self.model_path / "config.json"
        try:
            config = json.loads(config_path.read_text())
        except (OSError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return {
                "context_limit": unavailable(str(config_path), reason),
                "architecture": unavailable(str(config_path), reason),
            }
        # A multimodal checkpoint keeps the language model's shape under text_config,
        # so reading only the top level finds model_type and nothing else, and the
        # context limit is lost even though the checkpoint declares it.
        language = config.get("text_config")
        if not isinstance(language, dict):
            language = {}

        def declared(key: str) -> Any:
            value = config.get(key)
            return language.get(key) if value is None else value

        architecture = {
            key: declared(key)
            for key in (
                "model_type",
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
            )
            if declared(key) is not None
        }
        # A hybrid model keeps a growing cache only on its full-attention layers, so
        # the split is part of the shape rather than a detail of the layer count.
        layer_types = declared("layer_types")
        if isinstance(layer_types, list) and layer_types:
            architecture["layer_types"] = layer_types
        if declared("full_attention_interval") is not None:
            architecture["full_attention_interval"] = declared("full_attention_interval")
        dtype = declared("dtype") or declared("torch_dtype")
        if dtype is not None:
            architecture["dtype"] = dtype
        limit_key = (
            "max_position_embeddings"
            if config.get("max_position_embeddings") is not None
            else "text_config.max_position_embeddings"
        )
        return {
            "context_limit": fact(
                declared("max_position_embeddings"),
                grade=GRADE_DECLARED,
                source=f"{config_path}:{limit_key}",
            ),
            "architecture": fact(architecture, grade=GRADE_DECLARED, source=str(config_path)),
        }

    def _context_growth(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """State the rule that turns per-step text into a context-growth estimate.

        The real prompt is assembled inside the framework, so this is an
        explicit rule over declared values, never a measurement.
        """
        steps = self._steps(scenario.get("steps"))
        text = self._scenario_text(scenario.get("init_config"), scenario.get("steps"))
        per_step = text["tokens_max"].get("value")
        total = steps.get("total_steps")
        if per_step is None or not total:
            return unavailable(
                "init/steps.yaml",
                "per-step text length or step count is unknown; growth cannot be estimated",
            )
        return fact(
            int(per_step) * int(total),
            grade=GRADE_INFERRED,
            source="init/steps.yaml + scenario text token count",
            note=(
                "Rule: longest scenario prompt text x total step count. This assumes the "
                "framework carries earlier steps forward; it is an upper-bound shape "
                "indicator, not a measurement of the real assembled prompt."
            ),
        )

    def _arrival_facts(self, scenario: dict[str, Any]) -> dict[str, Any]:
        agents = self._agents(scenario.get("init_config"))
        steps = self._steps(scenario.get("steps"))
        count = len(agents)
        if not count:
            shape = unavailable("init_config.json", "agent count is unknown")
            peak = unavailable("init_config.json", "agent count is unknown")
        else:
            shape = fact(
                "step-synchronized burst",
                grade=GRADE_INFERRED,
                source="init/steps.yaml step structure + init_config.json agent list",
                note=(
                    "Rule: a step advances all agents together, so requests arrive in bursts "
                    "at each step boundary rather than as a steady stream."
                ),
            )
            peak = fact(
                count,
                grade=GRADE_INFERRED,
                source="init_config.json agent count",
                note=(
                    "Rule: calls within one agent are sequential; each agent can issue one request at a time. "
                    "one step; the bound assumes they are issued concurrently across agents."
                ),
            )
        return {
            "shape": shape,
            "peak_concurrent_requests": peak,
            "steps_per_episode": fact(
                steps.get("total_steps"),
                grade=GRADE_DECLARED if steps else GRADE_UNAVAILABLE,
                source="init/steps.yaml",
            ),
        }

    def _history_facts(self) -> dict[str, Any]:
        traces = [path for path in self.history_paths if Path(path).exists()]
        if not traces:
            return {
                "available": fact(
                    False,
                    grade=GRADE_UNAVAILABLE,
                    source="previous run artifacts",
                    note=(
                        "No earlier run artifacts were supplied. This is expected for a "
                        "first Campaign, so the profile must stand on code and configuration."
                    ),
                ),
                "traces": fact([], grade=GRADE_UNAVAILABLE, source="previous run artifacts"),
            }
        return {
            "available": fact(True, grade=GRADE_DECLARED, source="previous run artifacts"),
            "traces": fact(
                [str(path) for path in traces],
                grade=GRADE_DECLARED,
                source="previous run artifacts",
            ),
        }

    # -------------------------------------------------------------- tokenizer

    def _token_counts(self, texts: list[str]) -> dict[str, Any]:
        """Count tokens with the served model's real tokenizer when possible."""
        tokenizer, error = self._load_tokenizer()
        if tokenizer is not None:
            try:
                counts = [len(tokenizer(text)["input_ids"]) for text in texts]
            except Exception as exc:  # pragma: no cover - tokenizer-specific
                error = f"{type(exc).__name__}: {exc}"
            else:
                return fact(
                    counts,
                    grade=GRADE_MEASURED,
                    source=f"tokenizer at {self.model_path}",
                )
        executable = os.environ.get("EVOTAILOR_TOKENIZER_PYTHON")
        if executable and self.model_path:
            command = [
                executable,
                "-c",
                "import sys,json; from transformers import AutoTokenizer; p=json.load(sys.stdin); t=AutoTokenizer.from_pretrained(p['model'],local_files_only=True); print(json.dumps([len(t(x)['input_ids']) for x in p['texts']]))",
            ]
            environment = {
                k: v
                for k, v in os.environ.items()
                if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
            }
            completed = subprocess.run(
                command,
                input=json.dumps({"model": str(self.model_path), "texts": texts}),
                text=True,
                capture_output=True,
                timeout=120,
                env=environment,
                check=False,
            )
            if completed.returncode == 0:
                values = json.loads(completed.stdout)
                if len(values) != len(texts) or any(
                    not isinstance(v, int) or v <= 0 for v in values
                ):
                    raise ValueError("Served tokenizer produced invalid token counts")
                return fact(
                    values,
                    grade=GRADE_MEASURED,
                    source=f"{executable}: tokenizer at {self.model_path}",
                )
            error = f"Tokenizer subprocess failed: {completed.stderr[-500:]}"
        return unavailable(f"tokenizer at {self.model_path}", str(error))

    #: A non-empty probe a working tokenizer must split into at least one token.
    _TOKENIZER_PROBE = "The quick brown fox jumps over the lazy dog."

    @classmethod
    def _tokenizer_defect(cls, tokenizer: Any) -> str | None:
        """Return why ``tokenizer`` cannot produce real counts, or None if it can.

        ``transformers`` will happily resolve a tokenizer out of its local cache
        for a directory that only holds ``config.json``, and the result encodes
        every string to zero tokens. Reporting that as a measurement would be a
        fabricated number wearing a measured label, so the tokenizer is probed
        before any of its output is trusted.
        """
        try:
            vocabulary = int(getattr(tokenizer, "vocab_size", 0) or 0)
        except (TypeError, ValueError):
            vocabulary = 0
        if vocabulary <= 0:
            return "tokenizer reports an empty vocabulary"
        try:
            encoded = tokenizer(cls._TOKENIZER_PROBE)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        if not (encoded or {}).get("input_ids"):
            return "tokenizer produced no tokens for a non-empty probe"
        return None

    def _load_tokenizer(self) -> tuple[Any | None, str]:
        if self.model_path is None:
            return None, "no served model path"
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            return None, f"transformers unavailable: {exc}"
        # The host has no route to a model hub; every load must be local.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"
        defect = self._tokenizer_defect(tokenizer)
        if defect:
            return None, f"{self.model_path} did not yield a usable tokenizer: {defect}"
        return tokenizer, ""
