"""Exclusive leases for physical GPUs and serving endpoints.

Two scopes are needed, and conflating them caused real resource conflicts:

* a **Trial** scope, taken by the Trial wrapper while a service is running, so
  two deployments never share a card; and
* a **Campaign** scope, held from the moment a Campaign starts until its search
  Trials *and* its final-validation repeats are done.

The Trial scope alone is not enough. It is released between Trials, so a second
Campaign can observe a momentarily idle card, claim it, and then collide with
the first Campaign's next Trial. The Campaign scope is what makes "one Campaign
per GPU" true for the whole run rather than per deployment.

The Campaign lease is recorded in the run directory, so a resumed Campaign
re-takes it and an abandoned one can be identified by its holder file.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path


def capture_resource_snapshot(runtime: dict) -> dict:
    """Refresh live resources before selection; unavailable values remain explicit.

    Non-NVIDIA runtimes may supply resource_probe_command, which must emit a JSON
    object with gpu_count, gpu_total_bytes, and gpu_free_bytes (per-device minima).
    No measured resource state is inferred from a paper's example envelope.
    """
    import shutil
    import subprocess

    snapshot = {"timestamp": time.time(), "available": False,
                "cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()}
    try:
        snapshot["ram_available_bytes"] = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        snapshot["ram_available_bytes"] = None
    command = runtime.get("resource_probe_command")
    if command:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
            measured = json.loads(result.stdout)
            if not isinstance(measured, dict):
                raise ValueError("Resource probe must return a JSON object")
            snapshot.update(measured, available=True, source="configured_probe")
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            snapshot["unavailable_reason"] = str(exc)
        return snapshot
    if not shutil.which("nvidia-smi"):
        snapshot["unavailable_reason"] = "No resource probe configured and nvidia-smi is unavailable"
        return snapshot
    command = ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.free", "--format=csv,noheader,nounits"]
    devices = runtime.get("gpu_uuids") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if devices:
        command.append("--id=" + (",".join(devices) if isinstance(devices, list) else str(devices)))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
        snapshot.update(available=True, source="nvidia-smi", gpu_count=len(rows),
                        gpu_total_bytes=min(int(float(row[1])) for row in rows) * 1024**2,
                        gpu_free_bytes=min(int(float(row[2])) for row in rows) * 1024**2)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        snapshot["unavailable_reason"] = str(exc)
    return snapshot


def _lock_path(resource: str) -> Path:
    return Path("/tmp") / (
        "evotailor-resource-" + hashlib.sha256(resource.encode()).hexdigest() + ".lock"
    )


def _resources(gpu_uuids, endpoint: str | None) -> set[str]:
    resources = {f"gpu:{value}" for value in gpu_uuids}
    if endpoint:
        resources.add(f"endpoint:{endpoint}")
    return resources


@contextmanager
def serving_lease(gpu_uuids: list[str], endpoint: str | None, inherited=()):
    """Trial-scoped lease: one deployment per GPU and endpoint at a time.

    ``inherited`` names resources the caller's own Campaign already holds. A
    subprocess cannot inherit a flock, so without this a Trial would be refused
    the very card its own Campaign reserved. The caller is responsible for
    proving ownership; see the wrapper, which checks the campaign id.
    """
    skip = set(inherited)
    with ExitStack() as stack:
        for resource in sorted(_resources(gpu_uuids, endpoint) - skip):
            path = _lock_path(resource)
            handle = stack.enter_context(path.open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Serving resource is already in use: {resource}") from exc
        yield


class CampaignLeaseUnavailable(RuntimeError):
    """A Campaign could not take its GPU for the whole run."""


class CampaignResourceLease:
    """A Campaign-scoped hold on its GPUs and endpoint.

    Deliberately not a context manager used inside one Trial: it is meant to
    live for the whole Campaign, and to be released only when the Campaign has
    finished its search Trials and its repeats, or has failed terminally.

    The holder is written next to the lock so a blocked Campaign can say who
    holds the card instead of only that it is busy.
    """

    def __init__(self, gpu_uuids: list[str], endpoint: str | None, run_dir: Path, campaign_id: str):
        self.resources = sorted(_resources(gpu_uuids, endpoint))
        self.run_dir = Path(run_dir)
        self.campaign_id = campaign_id
        self._handles: list = []

    def acquire(self) -> "CampaignResourceLease":
        acquired: list = []
        try:
            for resource in self.resources:
                path = _lock_path(resource)
                handle = path.open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    handle.close()
                    raise CampaignLeaseUnavailable(
                        f"Campaign {self.campaign_id} cannot hold {resource}: "
                        f"another Campaign owns it ({self._holder(resource)})"
                    ) from exc
                acquired.append(handle)
        except Exception:
            for handle in acquired:
                try:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                finally:
                    handle.close()
            raise
        self._handles = acquired
        self._write_holder()
        return self

    def _holder(self, resource: str) -> str:
        path = _lock_path(resource)
        try:
            for line in path.read_text().splitlines():
                if line.strip():
                    return line.strip()
        except OSError:
            pass
        return "unknown holder"

    def _write_holder(self) -> None:
        stamp = json.dumps(
            {
                "campaign_id": self.campaign_id,
                "pid": os.getpid(),
                "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "resources": self.resources,
            }
        )
        for resource in self.resources:
            try:
                _lock_path(resource).write_text(stamp + "\n")
            except OSError:
                pass
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "campaign_resource_lease.json").write_text(
                json.dumps({"resources": self.resources, "holder": json.loads(stamp)}, indent=2)
                + "\n"
            )
        except OSError:
            pass

    def release(self) -> None:
        for resource in self.resources:
            try:
                _lock_path(resource).write_text("")
            except OSError:
                pass
        for handle in self._handles:
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                handle.close()
        self._handles = []

    def __enter__(self) -> "CampaignResourceLease":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
