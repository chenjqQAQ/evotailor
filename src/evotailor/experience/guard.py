"""The parts of the system an evolving agent may never rewrite.

EvoTailor measurement gates, immutable experiment facts, dataset admission and
verification criteria are human-controlled. An evolving agent must not rewrite
the checks that determine whether its own output is valid.

Everything else in the repository is fair game for a proposal. The boundary is
enforced by path before a patch is applied, not by asking the model nicely.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: Paths that decide whether a result is real. A patch touching any of them is
#: refused, and the refusal is recorded rather than silently dropped.
ROOT_TRUST_PATHS: tuple[str, ...] = (
    "src/evotailor/selection/preflight.py",
    "scripts/trial_wrapper.py",
    "scripts/agentsociety_runner.py",
    "scripts/aiconfigurator_bridge.py",
    "scripts/trace_replay_runner.py",
    "src/evotailor/execution/backends.py",
    "src/evotailor/contracts.py",
    "src/evotailor/objective.py",
    "src/evotailor/storage.py",
    "src/evotailor/experiment/__init__.py",
    "src/evotailor/selection/resources.py",
    "src/evotailor/experience/guard.py",
    "src/evotailor/experience/performance.py",
    "src/evotailor/loops/history.py",
    "pyproject.toml",
)


_DIFF_PATH = re.compile(r"^(?:\+\+\+|---)\s+(?:[ab]/)?(\S+)")


def is_root_trust(path: str) -> bool:
    """True when ``path`` is one of the files that decide whether a result is real."""
    normalised = path.strip().lstrip("./")
    if normalised == "tests" or normalised.startswith("tests/"):
        return True
    return any(
        normalised == trusted or normalised.endswith(f"/{trusted}") for trusted in ROOT_TRUST_PATHS
    )


def patch_paths(patch_text: str) -> list[str]:
    """Every file path a unified diff touches, in order of first appearance."""
    found: list[str] = []
    for line in patch_text.splitlines():
        match = _DIFF_PATH.match(line)
        if not match:
            continue
        path = match.group(1)
        if path == "/dev/null":
            continue
        if path not in found:
            found.append(path)
    return found


@dataclass
class Verification:
    ok: bool
    detail: str
    commit: str | None = None
    paths: list[str] = field(default_factory=list)


class WorkspaceGuard:
    """Apply a patch to the repository, verify it, and undo it if verification fails.

    Human review is replaced by a mechanical check: the patch is applied to a
    clean tree, the whole test suite must pass, and the change is committed only
    when it does. A patch that breaks the suite is reversed with the same patch
    rather than with a broad checkout, so unrelated work cannot be discarded.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        test_command: list[str] | None = None,
        timeout_s: int = 3600,
        performance_command: list[str] | None = None,
        min_repeats: int = 3,
        isolated: bool = True,
        performance_timeout_s: int = 28800,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.test_command = test_command or [
            ".venv/bin/python",
            "-m",
            "pytest",
            "-m",
            "not live_model",
            "-q",
        ]
        self.timeout_s = timeout_s
        self.performance_command = performance_command or []
        self.min_repeats = min_repeats
        self.isolated = isolated
        self.performance_timeout_s = performance_timeout_s

    def _git(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout_s,
        )

    def is_clean(self) -> tuple[bool, str]:
        status = self._git("status", "--porcelain")
        if status.returncode:
            return False, f"git status failed: {status.stderr.strip()[:200]}"
        if status.stdout.strip():
            return False, "the working tree has uncommitted changes"
        return True, ""

    def apply_and_verify(self, patch_text: str, *, message: str) -> Verification:
        if self.isolated:
            return self._isolated_apply(patch_text, message=message)
        paths = patch_paths(patch_text)
        if not paths:
            return Verification(False, "the patch touches no file")
        blocked = sorted(path for path in paths if is_root_trust(path))
        if blocked:
            return Verification(
                False,
                "the patch touches the root-trust layer that decides whether a result "
                f"is real, so it is refused: {blocked}",
                paths=paths,
            )

        clean, reason = self.is_clean()
        if not clean:
            return Verification(
                False, f"refusing to apply a patch onto a dirty tree: {reason}", paths=paths
            )

        # The patch file must live outside the repository: ``git add -A`` would
        # otherwise commit the temporary file itself and leave the tree dirty.
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
            patch_file = Path(handle.name)
            handle.write(patch_text)
        try:
            check = self._git("apply", "--check", str(patch_file))
            if check.returncode:
                return Verification(
                    False,
                    f"the patch does not apply: {check.stderr.strip()[:400]}",
                    paths=paths,
                )
            applied = self._git("apply", str(patch_file))
            if applied.returncode:
                return Verification(
                    False,
                    f"git apply failed: {applied.stderr.strip()[:400]}",
                    paths=paths,
                )
            try:
                tests = subprocess.run(
                    self.test_command,
                    cwd=self.repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=self.timeout_s,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                reversal = self._git("apply", "-R", str(patch_file))
                if reversal.returncode or not self.is_clean()[0]:
                    raise RuntimeError(
                        "verification failed and rollback did not restore a clean tree"
                    ) from exc
                return Verification(
                    False,
                    f"verification could not complete; patch reversed: {type(exc).__name__}: {exc}",
                    paths=paths,
                )
            if tests.returncode:
                reversal = self._git("apply", "-R", str(patch_file))
                clean_again, _ = self.is_clean()
                if reversal.returncode or not clean_again:
                    raise RuntimeError("test failure rollback did not restore a clean tree")
                return Verification(
                    False,
                    "the test suite failed, so the patch was reversed: "
                    + (tests.stdout or tests.stderr).strip()[-600:],
                    paths=paths,
                )
            self._git("add", "-A")
            committed = self._git(
                "-c",
                "user.name=EvoTailor",
                "-c",
                "user.email=evotailor@local",
                "commit",
                "-q",
                "-m",
                message,
            )
            if committed.returncode:
                self._git("reset")
                reversal = self._git("apply", "-R", str(patch_file))
                if reversal.returncode or not self.is_clean()[0]:
                    raise RuntimeError("commit failure rollback did not restore a clean tree")
                return Verification(
                    False,
                    f"the patch passed its tests but could not be committed: "
                    f"{committed.stderr.strip()[:200]}",
                    paths=paths,
                )
            revision = self._git("rev-parse", "--short", "HEAD")
            return Verification(
                True,
                "applied, verified, and committed",
                commit=revision.stdout.strip() or None,
                paths=paths,
            )
        finally:
            patch_file.unlink(missing_ok=True)

    def _isolated_apply(self, patch_text: str, *, message: str) -> Verification:
        import json
        import os

        from evotailor.experience.performance import assess

        paths = patch_paths(patch_text)
        if not self.performance_command:
            return Verification(
                False, "No real performance evaluator configured; patch not adopted", paths=paths
            )
        source_status = self._git(
            "status", "--porcelain", "--", "src", "scripts", "tests", "pyproject.toml"
        )
        if source_status.returncode or source_status.stdout.strip():
            return Verification(
                False,
                "Source must be committed; unrelated configuration changes preserved",
                paths=paths,
            )
        with tempfile.TemporaryDirectory(prefix="evotailor-evolution-") as directory:
            candidate = Path(directory) / "candidate"
            added = self._git("worktree", "add", "--detach", str(candidate), "HEAD")
            if added.returncode:
                return Verification(False, added.stderr[-500:], paths=paths)
            try:
                command = [
                    str((self.repo_root / x).resolve()) if x.startswith(".venv/") else x
                    for x in self.test_command
                ]
                guard = WorkspaceGuard(
                    candidate, test_command=command, timeout_s=self.timeout_s, isolated=False
                )
                # Make the candidate source win over the editable installation of the parent.
                original_pythonpath = os.environ.get("PYTHONPATH")
                os.environ["PYTHONPATH"] = str(candidate / "src")
                try:
                    result = guard.apply_and_verify(patch_text, message=message)
                finally:
                    if original_pythonpath is None:
                        os.environ.pop("PYTHONPATH", None)
                    else:
                        os.environ["PYTHONPATH"] = original_pythonpath
                if not result.ok:
                    return result
                report = Path(directory) / "performance.json"
                evaluation = subprocess.run(
                    [
                        *self.performance_command,
                        "--baseline-root",
                        str(self.repo_root),
                        "--candidate-root",
                        str(candidate),
                        "--output",
                        str(report),
                    ],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=self.performance_timeout_s,
                    check=False,
                )
                if evaluation.returncode or not report.is_file():
                    return Verification(
                        False, "Real evaluation failed: " + evaluation.stderr[-500:], paths=paths
                    )
                measured = json.loads(report.read_text())
                for label, expected in (("baseline", self.repo_root), ("candidate", candidate)):
                    if any(
                        Path(e["source_root"]).resolve() != expected.resolve()
                        for e in measured.get(label, [])
                    ):
                        return Verification(
                            False,
                            "Evaluator returned measurements for another source tree",
                            paths=paths,
                        )
                decision = assess(report, self.min_repeats)
                audit = (
                    self.repo_root
                    / "runs"
                    / "evolution"
                    / ("performance-" + result.commit + ".json")
                )
                audit.parent.mkdir(parents=True, exist_ok=True)
                audit.write_text(
                    json.dumps(
                        {"decision": decision, "report": json.loads(report.read_text())}, indent=2
                    )
                    + "\n"
                )
                if not decision["accepted"]:
                    return Verification(
                        False,
                        "Real evaluation did not improve with preserved success: " + str(audit),
                        paths=paths,
                    )
                adopted = self._git("cherry-pick", result.commit)
                if adopted.returncode:
                    self._git("cherry-pick", "--abort")
                    return Verification(
                        False,
                        "Candidate could not be adopted: " + adopted.stderr[-500:],
                        paths=paths,
                    )
                return Verification(
                    True,
                    "Functional tests and real paired search gate passed: " + str(audit),
                    commit=result.commit,
                    paths=paths,
                )
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
                return Verification(
                    False,
                    f"Real evolution validation could not complete: {type(exc).__name__}: {exc}",
                    paths=paths,
                )
            finally:
                self._git("worktree", "remove", "--force", str(candidate))
