from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from evotailor.spaces.designer import SpaceDesignAgent
from evotailor.config import load_config, load_dotenv
from evotailor.contracts import ExecutionBackend, RunMode
from evotailor.dashboard import DashboardServer
from evotailor.workloads.discovery.inspectors import write_discovery
from evotailor.spaces.engine_flags import ensure_passable
from evotailor.selection.engine_probe import probe_from_environment
from evotailor.experiment import atomic_json, discovery_identity
from evotailor.llm import client_from_config
from evotailor.selection.preflight import run_preflight
from evotailor.loops.runner import CampaignRunner
from evotailor.spaces.search_space import SearchSpace
from evotailor.tracing import ConsoleTrace

app = typer.Typer(no_args_is_help=True, help="EvoTailor campaign runner")


def _runtime_parameter_arguments() -> dict[str, Any]:
    """The mappings the runtime config already declares for candidate parameters."""
    path = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
    if not path:
        return {}
    try:
        payload = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    except (OSError, ValueError):
        return {}
    arguments = payload.get("parameter_arguments")
    return arguments if isinstance(arguments, dict) else {}


@app.command()
def discover(
    task_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Path = Path("runs/discovery.json"),
    vllm_binary: Annotated[str | None, typer.Option()] = None,
    gpu_probe_image: Annotated[
        str | None,
        typer.Option(help="Probe the allocated GPUs inside this image before falling back."),
    ] = None,
    model_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "The model that will actually be served. Its tokenizer and context limit "
                "are what make the context-length facts real measurements."
            )
        ),
    ] = None,
    scenario_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "The scenario to profile. Supply it when the task root holds several "
                "scenarios, so the profile cannot describe the wrong workload."
            )
        ),
    ] = None,
) -> None:
    load_dotenv(Path(".env"))
    result = write_discovery(
        output,
        task_root,
        vllm_binary,
        gpu_probe_image,
        model_path=model_path,
        scenario_path=scenario_path,
    )
    typer.echo(
        json.dumps(
            {"output": str(output), "gpu_probe": result["hardware"].get("gpus", {})},
            indent=2,
        )
    )


@app.command()
def prepare(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/default.yaml"),
    discovery: Annotated[Path, typer.Option(exists=True)] = Path("runs/discovery.json"),
    output: Path = Path("runs/prepared_search_space.json"),
    project_root: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """Use discovered facts once, before Trial 0, to extend the base space."""
    project_root = project_root.resolve()
    load_dotenv(project_root / ".env")
    campaign = load_config(config)
    base_path = Path(campaign.search_space)
    if not base_path.is_absolute():
        base_path = project_root / base_path
    client = client_from_config(campaign)

    discovery_payload = json.loads(discovery.read_text())
    prepared, decision = SpaceDesignAgent(client).design(
        SearchSpace.load(base_path),
        discovery_payload,
        campaign.max_trials,
    )
    prepared, derived_arguments, dropped = ensure_passable(
        prepared,
        SearchSpace.load(base_path),
        set((discovery_payload.get("vllm") or {}).get("supported_flags") or []),
        _runtime_parameter_arguments(),
        probe_from_environment(),
    )
    decision["derived_parameter_arguments"] = derived_arguments
    decision["dropped_dimensions"] = dropped
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(prepared.model_dump_json(indent=2) + "\n")
    decision_path = output.with_suffix(".decision.json")
    decision_path.write_text(json.dumps(decision, indent=2, default=str) + "\n")
    output.with_suffix(".arguments.json").write_text(
        json.dumps(derived_arguments, indent=2, ensure_ascii=False) + "\n"
    )
    typer.echo(json.dumps({"search_space": str(output), "decision": str(decision_path)}))


@app.command()
def run(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/default.yaml"),
    project_root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[
        bool,
        typer.Option("--json-output", help="Print the machine-readable final summary."),
    ] = False,
    dashboard: Annotated[
        bool,
        typer.Option("--dashboard/--no-dashboard", help="Run the live web dashboard."),
    ] = True,
    dashboard_host: Annotated[str, typer.Option()] = "0.0.0.0",
    dashboard_port: Annotated[int, typer.Option(min=0, max=65535)] = 8090,
    task_root: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="Inspect this task before Trial 1."),
    ] = None,
) -> None:
    project_root = project_root.resolve()
    load_dotenv(project_root / ".env")
    spec = load_config(config)
    run_dir = project_root / spec.run_root / spec.campaign_id
    run_dir.mkdir(parents=True, exist_ok=True)
    server = None
    if dashboard:
        server = DashboardServer(run_dir, host=dashboard_host, port=dashboard_port)
        server.start()
        typer.echo("\n" + "═" * 78)
        typer.echo(f"Live dashboard started at http://127.0.0.1:{server.port}")
        typer.echo("如果通过远程开发环境访问，请转发或打开上述端口。")
        typer.echo("面板会在本次运行结束时自动关闭。")
        typer.echo("═" * 78 + "\n")
    try:
        bootstrap_trace = ConsoleTrace(run_dir, spec.console_progress)
        if task_root is not None:
            discovery_path = run_dir / "discovery.json"
            prepared_path = run_dir / "prepared_search_space.json"
            decision_path = run_dir / "space_design_decision.json"
            derived_path = run_dir / "derived_parameter_arguments.json"
            runtime_path = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
            runtime = yaml.safe_load(Path(runtime_path).read_text()) if runtime_path else {}
            identity = discovery_identity(task_root, runtime)
            identity_path = run_dir / "discovery_identity.json"
            if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
                raise ValueError(
                    "Task, hardware/runtime or model inputs changed; use a new Campaign"
                )
            atomic_json(identity_path, identity)
            if prepared_path.exists() and discovery_path.exists() and decision_path.exists():
                discovery_payload = json.loads(discovery_path.read_text())
                decision = json.loads(decision_path.read_text())
                prepared = SearchSpace.load(prepared_path)
                reused = True
            else:
                if spec.run_mode == RunMode.BASELINE:
                    raise ValueError(
                        "Baseline cannot call a space-design model; supply the shared prepared search space"
                    )
                bootstrap_trace.emit(
                    "Project inspection",
                    "正在分析上层任务代码、硬件和当前 vLLM 能力。",
                    任务目录=str(task_root.resolve()),
                )
                runtime_path = os.environ.get("EVOTAILOR_RUNTIME_CONFIG")
                runtime = yaml.safe_load(Path(runtime_path).read_text()) if runtime_path else {}
                discovery_payload = write_discovery(
                    discovery_path,
                    task_root,
                    vllm_binary=(str(runtime["vllm_environment_path"]) + "/bin/vllm")
                    if runtime.get("vllm_environment_path")
                    else None,
                    gpu_probe_image=runtime.get("image"),
                    model_path=runtime.get("model_path"),
                    project_root=runtime.get("project_path"),
                    scenario_path=runtime.get("scenario_path"),
                )
                base_path = Path(spec.search_space)
                if not base_path.is_absolute():
                    base_path = project_root / base_path
                client = client_from_config(spec)
                client.check_connectivity()
                prepared, decision = SpaceDesignAgent(client).design(
                    SearchSpace.load(base_path), discovery_payload, spec.max_trials
                )
                # A designed dimension is only usable if the Trial wrapper can turn
                # it into a flag the installed vLLM accepts. Without this the
                # dimension survives into the frozen space and every Trial fails
                # before a container starts.
                probe = probe_from_environment()
                if probe is None:
                    bootstrap_trace.emit(
                        "Search space construction",
                        "未配置真实部署模型路径，Added dimensions只能核对 flag 是否被列出，"
                        "无法确认它在本平台真的能启动。",
                    )
                prepared, derived_arguments, dropped = ensure_passable(
                    prepared,
                    SearchSpace.load(base_path),
                    set((discovery_payload.get("vllm") or {}).get("supported_flags") or []),
                    _runtime_parameter_arguments(),
                    probe,
                )
                decision["derived_parameter_arguments"] = derived_arguments
                decision["dropped_dimensions"] = dropped
                if derived_arguments or dropped:
                    bootstrap_trace.emit(
                        "Search space construction",
                        "已记录Added dimensions的参数映射与真实Launch check证据。",
                        自动推导的映射=derived_arguments,
                        已丢弃的维度=dropped,
                    )
                derived_path.write_text(
                    json.dumps(derived_arguments, indent=2, ensure_ascii=False) + "\n"
                )
                prepared_path.write_text(prepared.model_dump_json(indent=2) + "\n")
                decision_path.write_text(
                    json.dumps(decision, indent=2, ensure_ascii=False, default=str) + "\n"
                )
                reused = False
            task = discovery_payload.get("task") or {}
            hardware = discovery_payload.get("hardware") or {}
            vllm = discovery_payload.get("vllm") or {}
            bootstrap_trace.emit(
                "Project features",
                "已完成运行前Project inspection。",
                Agent summary (natural language)=decision.get("project_summary", []),
                扫描文件数=task.get("file_count_sampled"),
                可能的启动入口=(task.get("entrypoint_candidates") or [])[:8],
                CPU核心数=hardware.get("cpu_count"),
                GPU信息=(hardware.get("gpus") or {}).get("stdout", "unavailable").strip(),
                vLLM版本=(vllm.get("version") or {}).get("stdout", "unavailable").strip(),
                支持的vLLM参数数=len(vllm.get("supported_flags") or []),
            )
            base_path = Path(spec.search_space)
            if not base_path.is_absolute():
                base_path = project_root / base_path
            base = SearchSpace.load(base_path)
            added = sorted(set(prepared.dimensions) - set(base.dimensions))
            bootstrap_trace.emit(
                "Search space construction",
                "Agent 已在 Trial 0 之前完成搜索空间决策，之后将保持冻结。",
                Base dimension count=len(base.dimensions),
                Final dimension count=len(prepared.dimensions),
                Added dimensions=added,
                是否复用已有决策=reused,
                Decision rationale=decision.get("rationale"),
                证据=decision.get("evidence", []),
            )
            spec = spec.model_copy(
                update={
                    "search_space": str(prepared_path),
                    "discovery_artifact": str(discovery_path),
                }
            )
        else:
            space_path = Path(spec.search_space)
            if not space_path.is_absolute():
                space_path = project_root / space_path
            base = SearchSpace.load(space_path)
            bootstrap_trace.emit(
                "Project inspection",
                "本次运行没有提供 --task-root，因此未执行上层任务代码与硬件探查。",
                如何启用="运行时增加 --task-root /上层任务代码目录",
            )
            bootstrap_trace.emit(
                "Search space construction",
                "本次直接使用基础搜索空间，没有调用 Agent 增加维度。",
                Base dimension count=len(base.dimensions),
                Final dimension count=len(base.dimensions),
                Added dimensions=[],
            )
        if spec.execution_backend != ExecutionBackend.SIMULATED:
            check = run_preflight(spec, project_root)
            (run_dir / "preflight.json").write_text(json.dumps(check, indent=2) + "\n")
            if not check["ok"]:
                raise ValueError(f"Real execution preflight failed: {check['issues']}")
        try:
            summary = CampaignRunner(spec, project_root).run()
        except Exception as exc:
            bootstrap_trace.emit(
                "Task launch失败",
                "Campaign 在进入 Trial 前失败，面板保留本次失败状态。",
                错误类型=type(exc).__name__,
                错误=str(exc),
                结果目录=str(run_dir),
                建议="更换 campaign_id 或清理/复用匹配的 frozen_search_space.json 后重新运行。",
            )
            raise
    finally:
        if server is not None:
            server.stop()
    if json_output or not spec.console_progress:
        typer.echo(json.dumps(summary, indent=2, default=str))
    else:
        typer.echo(
            "Run finished. Full results saved to:"
            f"{project_root / spec.run_root / spec.campaign_id / 'summary.json'}"
        )


@app.command()
def preflight(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/default.yaml"),
    project_root: Annotated[Path, typer.Option()] = Path("."),
    runtime_config: Annotated[Path | None, typer.Option(exists=True)] = None,
) -> None:
    project_root = project_root.resolve()
    load_dotenv(project_root / ".env")
    if runtime_config is not None:
        import os

        os.environ["EVOTAILOR_RUNTIME_CONFIG"] = str(runtime_config.resolve())
    result = run_preflight(load_config(config), project_root)
    typer.echo(json.dumps(result, indent=2, default=str))
    if not result["ok"]:
        raise typer.Exit(code=2)


@app.command()
def replay(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    dashboard_host: Annotated[str, typer.Option()] = "0.0.0.0",
    dashboard_port: Annotated[int, typer.Option(min=0, max=65535)] = 8090,
) -> None:
    """Open an immutable replay dashboard for a completed or partial Campaign."""
    run_dir = run_dir.resolve()
    if not (run_dir / "trace.jsonl").exists():
        raise typer.BadParameter("run_dir does not contain trace.jsonl")
    server = DashboardServer(run_dir, host=dashboard_host, port=dashboard_port, replay=True)
    server.start()
    typer.echo(f"Replay dashboard started at http://127.0.0.1:{server.port}")
    typer.echo("拖动面板中的回放进度条可查看任意历史步骤；按 Ctrl-C 退出。")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


@app.command()
def report(
    run_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
) -> None:
    typer.echo((run_dir / "summary.json").read_text())


if __name__ == "__main__":
    app()
