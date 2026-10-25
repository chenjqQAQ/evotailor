"""Bounded source context for real patches; never expose environment files or keys."""

import hashlib
from pathlib import Path

FILES = (
    "src/evotailor/loops/proposals.py",
    "src/evotailor/selection/agent.py",
    "src/evotailor/loops/episode_planner.py",
    "src/evotailor/selection/prediction/configuration.py",
    "src/evotailor/loops/policy.py",
    "tests/test_paper_contracts.py",
    "tests/test_spaces.py",
)


def source_context(root):
    result = []
    for name in FILES:
        path = Path(root) / name
        if path.is_file():
            raw = path.read_bytes()
            result.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "source": raw.decode(),
                    "complete": True,
                }
            )
    revision = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return {
        "files": result,
        "patch_scope": "Only propose exact patches to supplied source files; do not invent unseen code.",
        "repo_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "revision_available": revision.returncode == 0,
    }
