from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable, Iterable

from . import __version__
from .formatting import (
    Column,
    color_for_state,
    colorize,
    format_mb,
    format_ratio,
    join_short,
    print_table,
    should_color,
    split_csv,
)
from .slurm import (
    HistoryJob,
    Job,
    Node,
    SlurmError,
    current_user,
    expand_nodelist,
    gpu_model_from_feature,
    gpu_model_from_node,
    sacct_history,
    scontrol_nodes,
    sinfo_partitions,
    squeue_jobs,
)

ISO_DATE_RE = re.compile(r"^\d{4}-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Output JSON for jq or scripts.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    parser.add_argument("--watch", type=positive_int, metavar="SEC", help="Refresh every SEC seconds.")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_watch(args: argparse.Namespace, render: Callable[[], None]) -> int:
    if not getattr(args, "watch", None):
        render()
        return 0
    try:
        while True:
            print("\x1b[2J\x1b[H", end="")
            render()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


def job_gpu_summary(
    tres_or_gres: str, feature: str | None = None, node_models: Iterable[str] | None = None
) -> str:
    if not tres_or_gres or tres_or_gres == "-":
        return "-"
    typed: list[str] = []
    generic: str | None = None
    for item in tres_or_gres.split(","):
        item = item.strip()
        if item.startswith("gres/gpu:") and "=" in item:
            key, value = item.split("=", 1)
            typed.append(f"{key.removeprefix('gres/gpu:')}={value}")
        elif item.startswith("gres/gpu="):
            generic = item.split("=", 1)[1]
    if typed:
        return ",".join(typed)
    if generic is not None:
        model = gpu_model_from_feature(feature)
        models = sorted({item for item in node_models or [] if item})
        if model is None and len(models) == 1:
            model = models[0]
        if model:
            return f"{model}={generic}"
        if models:
            shown = "/".join(models[:2])
            suffix = f"+{len(models) - 2}" if len(models) > 2 else ""
            return f"gpu={generic}({shown}{suffix})"
        return f"gpu={generic}"
    if tres_or_gres.startswith("gpu"):
        return tres_or_gres
    return "-"


def job_has_generic_gpu(tres_or_gres: str) -> bool:
    if not tres_or_gres or tres_or_gres == "-":
        return False
    has_generic = False
    for item in tres_or_gres.split(","):
        item = item.strip()
        if item.startswith("gres/gpu:") and "=" in item:
            return False
        if item.startswith("gres/gpu="):
            has_generic = True
    return has_generic


def job_is_pending(job: Job) -> bool:
    return job.state.upper().startswith(("PENDING", "PD"))


def job_candidate_nodes(job: Job) -> list[str]:
    candidates: list[str] = []
    candidates.extend(expand_nodelist(job.sched_nodes))
    if not job_is_pending(job):
        candidates.extend(expand_nodelist(job.location))
    seen: set[str] = set()
    return [node for node in candidates if not (node in seen or seen.add(node))]


def job_needs_node_gpu_lookup(job: Job) -> bool:
    if not job_has_generic_gpu(job.gres):
        return False
    if gpu_model_from_feature(job.feature):
        return False
    return bool(job_candidate_nodes(job))


def load_node_models_for_jobs(jobs: Iterable[Job]) -> dict[str, list[str]]:
    jobs_to_enrich = [job for job in jobs if job_needs_node_gpu_lookup(job)]
    if not jobs_to_enrich:
        return {}
    try:
        nodes = {node.name: node for node in scontrol_nodes()}
    except SlurmError:
        return {}

    result: dict[str, list[str]] = {}
    for job in jobs_to_enrich:
        models: list[str] = []
        for node_name in job_candidate_nodes(job):
            node = nodes.get(node_name)
            if not node:
                continue
            model = gpu_model_from_node(node)
            if model:
                models.append(model)
        if models:
            result[job.job_id] = models
    return result


def format_start_time(value: str) -> str:
    if not value or value in {"-", "N/A", "Unknown", "INVALID"}:
        return "-"
    match = ISO_DATE_RE.match(value)
    if not match:
        return value
    month, day, hour, minute = match.groups()
    if hour and minute:
        return f"{month}-{day} {hour}:{minute}"
    return f"{month}-{day}"


def color_ratio_value(value: str, free: int | None, total: int | None, enabled: bool) -> str:
    if free is None or total is None or total <= 0:
        return colorize(value, "gray", enabled)
    if free <= 0:
        return colorize(value, "red", enabled)
    if free < total:
        return colorize(value, "yellow", enabled)
    return colorize(value, "green", enabled)


def color_resource_value(value: str, enabled: bool) -> str:
    if not value or value == "-":
        return colorize("-", "gray", enabled)
    return colorize(value, "cyan", enabled)


def job_to_row(
    job: Job, *, color: bool, long: bool, node_models: Iterable[str] | None = None
) -> dict[str, str]:
    state = colorize(job.state, color_for_state(job.state), color)
    gpu = job_gpu_summary(job.gres, job.feature, node_models)
    is_pending = job.state.upper().startswith(("PENDING", "PD"))
    row = {
        "job_id": colorize(job.job_id, "bold", color),
        "name": job.name,
        "state": state,
        "partition": colorize(job.partition, "cyan", color),
        "location": colorize(job.location, "yellow" if is_pending else "blue", color),
        "gpu": color_resource_value(gpu, color),
        "cpus": color_resource_value(job.cpus, color),
        "memory": color_resource_value(job.memory, color),
        "nodes": job.nodes,
        "elapsed": job.elapsed,
        "limit": job.time_limit,
        "start": colorize(format_start_time(job.start_time), "gray" if is_pending else None, color),
    }
    if long:
        row.update({"submit": job.submit_time, "user": job.user, "priority": job.priority})
    return row


def add_queue_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=current_user(), help="Query this user. Defaults to the current user.")
    parser.add_argument("-a", "--all", action="store_true", help="Show jobs from all users.")
    parser.add_argument("-p", "--partition", help="Only show jobs in this partition.")
    parser.add_argument(
        "-t",
        "--states",
        help="Filter Slurm states, for example R,PD or RUNNING,PENDING.",
    )
    parser.add_argument("-l", "--long", action="store_true", help="Show submit time, user, and priority.")
    add_common_output_flags(parser)


def command_queue(args: argparse.Namespace) -> int:
    states = split_csv(args.states)

    def fetch() -> list[Job]:
        return squeue_jobs(user=args.user, all_users=args.all, states=states, partition=args.partition)

    def render_once() -> None:
        jobs = fetch()
        if args.json:
            print_json([asdict(job) for job in jobs])
            return
        color = should_color(args.no_color)
        node_models = load_node_models_for_jobs(jobs)
        rows = [
            job_to_row(job, color=color, long=args.long, node_models=node_models.get(job.job_id))
            for job in jobs
        ]
        if not rows:
            who = "all users" if args.all else args.user
            print(f"No Slurm jobs found for {who}.")
            return
        columns = [
            Column("job_id", "JOBID", min_width=6, max_width=14),
            Column("name", "NAME", min_width=8, max_width=24),
            Column("state", "STATE", min_width=7, max_width=13),
            Column("partition", "PART", min_width=5, max_width=12),
            Column("nodes", "N", min_width=1, max_width=4, align="right"),
            Column("location", "WHERE", min_width=5, max_width=28),
            Column("gpu", "GPU", min_width=3, max_width=20),
            Column("cpus", "CPU", min_width=3, max_width=5, align="right"),
            Column("memory", "MEM", min_width=3, max_width=8, align="right"),
            Column("elapsed", "TIME", min_width=4, max_width=12, align="right"),
            Column("limit", "LIMIT", min_width=5, max_width=12, align="right"),
            Column("start", "START", min_width=5, max_width=11),
        ]
        if args.long:
            columns.extend(
                [
                    Column("submit", "SUBMIT", min_width=6, max_width=19),
                    Column("user", "USER", min_width=4, max_width=14),
                    Column("priority", "PRIO", min_width=4, max_width=10, align="right"),
                ]
            )
        print_table(rows, columns)

    return run_watch(args, render_once)


def node_gpu_type_summary(node: Node, *, color: bool) -> str:
    typed = node.gpu_types
    if typed:
        return ", ".join(
            f"{colorize(name, 'cyan', color)} {color_ratio_value(f'{free}/{total}', free, total, color)}"
            for name, (free, total) in sorted(typed.items())
        )
    total = node.gpu_total
    free = node.gpu_free
    if total is None:
        return colorize("-", "gray", color)
    return f"{colorize('gpu', 'cyan', color)} {color_ratio_value(format_ratio(free, total), free, total, color)}"


def node_to_dict(node: Node) -> dict[str, Any]:
    data = asdict(node)
    data.update(
        {
            "cpus_free": node.cpus_free,
            "mem_free_sched_mb": node.mem_free_sched_mb,
            "gpu_total": node.gpu_total,
            "gpu_alloc": node.gpu_alloc,
            "gpu_free": node.gpu_free,
            "gpu_types": {
                key: {"free": value[0], "total": value[1]}
                for key, value in node.gpu_types.items()
            },
            "allocatable_now": node.is_allocatable_now,
        }
    )
    return data


def node_to_row(node: Node, *, color: bool) -> dict[str, str]:
    state = colorize(node.state, color_for_state(node.state), color)
    mem_text = f"{format_mb(node.mem_free_sched_mb)}/{format_mb(node.mem_total_mb)}"
    return {
        "name": colorize(node.name, "bold", color),
        "partition": colorize(",".join(node.partitions) or "-", "cyan", color),
        "gpu": node_gpu_type_summary(node, color=color),
        "cpu": color_ratio_value(format_ratio(node.cpus_free, node.cpus_total), node.cpus_free, node.cpus_total, color),
        "mem": color_ratio_value(mem_text, node.mem_free_sched_mb, node.mem_total_mb, color),
        "free_mem": colorize(format_mb(node.mem_free_os_mb), "blue", color),
        "state": state,
        "features": colorize(node.features, "blue", color),
    }


def filter_nodes(
    nodes: Iterable[Node],
    *,
    partition: str | None = None,
    gpu_type: str | None = None,
    min_gpus: int = 1,
    available_only: bool = False,
    gpu_only: bool = False,
    states: list[str] | None = None,
) -> list[Node]:
    result: list[Node] = []
    wanted_states = [state.upper() for state in states or []]
    for node in nodes:
        if partition and partition not in node.partitions:
            continue
        if wanted_states and not any(state in node.state.upper() for state in wanted_states):
            continue
        if gpu_type:
            if gpu_type not in node.gpu_types and gpu_type not in node.gres:
                continue
        if gpu_only and (node.gpu_total or 0) <= 0:
            continue
        if available_only:
            if not node.is_allocatable_now:
                continue
            if (node.gpu_free or 0) < min_gpus:
                continue
            if gpu_type and (node.gpu_types.get(gpu_type, (0, 0))[0] < min_gpus):
                continue
        result.append(node)
    return sorted(result, key=lambda item: (",".join(item.partitions), item.name))


def add_node_filter_args(parser: argparse.ArgumentParser, *, include_avail_defaults: bool = False) -> None:
    parser.add_argument("-p", "--partition", help="Only show nodes in this partition.")
    parser.add_argument("-g", "--gpu-type", help="Only show nodes with this GPU type, for example a100.")
    parser.add_argument("--min-gpus", type=int, default=1, help="Minimum free GPU count. Defaults to 1.")
    parser.add_argument(
        "--states",
        help="Filter node states, for example IDLE,MIXED,DRAIN.",
    )
    if include_avail_defaults:
        parser.add_argument("--all", action="store_true", help="Show all GPU nodes, not only currently available nodes.")
    else:
        parser.add_argument("--available", action="store_true", help="Only show schedulable nodes with free GPUs.")
    add_common_output_flags(parser)


def command_avail(args: argparse.Namespace) -> int:
    states = split_csv(args.states)

    def render_once() -> None:
        all_matching_gpu_nodes = filter_nodes(
            scontrol_nodes(),
            partition=args.partition,
            gpu_type=args.gpu_type,
            min_gpus=args.min_gpus,
            available_only=False,
            gpu_only=True,
            states=states,
        )
        nodes = (
            all_matching_gpu_nodes
            if args.all
            else filter_nodes(
                all_matching_gpu_nodes,
                gpu_type=args.gpu_type,
                min_gpus=args.min_gpus,
                available_only=True,
                gpu_only=True,
            )
        )
        if args.json:
            print_json([node_to_dict(node) for node in nodes])
            return
        hidden = [node for node in all_matching_gpu_nodes if node not in nodes]
        if not nodes:
            print("No free GPU resources match the filters.")
            if hidden:
                print(
                    f"Hidden GPU nodes: {len(hidden)} "
                    f"({join_short([node.name for node in hidden], 8)}). Use `javail --all` to show all."
                )
            return
        color = should_color(args.no_color)
        rows = [node_to_row(node, color=color) for node in nodes]
        print_table(
            rows,
            [
                Column("name", "NODE", min_width=5, max_width=18),
                Column("partition", "PART", min_width=5, max_width=18),
                Column("cpu", "CPU", min_width=5, max_width=9, align="right"),
                Column("gpu", "GPU", min_width=8, max_width=28),
                Column("mem", "MEM", min_width=7, max_width=13, align="right"),
                Column("state", "STATE", min_width=5, max_width=16),
                Column("features", "FEATURES", min_width=5, max_width=26),
            ],
        )
        if hidden:
            print(
                colorize(
                    f"Hidden GPU nodes: {len(hidden)} "
                    f"({join_short([node.name for node in hidden], 8)}). Use `javail --all` to show all.",
                    "gray",
                    color,
                )
            )

    return run_watch(args, render_once)


def command_nodes(args: argparse.Namespace) -> int:
    states = split_csv(args.states)

    def render_once() -> None:
        nodes = filter_nodes(
            scontrol_nodes(),
            partition=args.partition,
            gpu_type=args.gpu_type,
            min_gpus=args.min_gpus,
            available_only=args.available,
            gpu_only=False,
            states=states,
        )
        if args.json:
            print_json([node_to_dict(node) for node in nodes])
            return
        if not nodes:
            print("No nodes match the filters.")
            return
        color = should_color(args.no_color)
        rows = [node_to_row(node, color=color) for node in nodes]
        print_table(
            rows,
            [
                Column("name", "NODE", min_width=5, max_width=18),
                Column("partition", "PART", min_width=5, max_width=20),
                Column("cpu", "CPU", min_width=5, max_width=9, align="right"),
                Column("gpu", "GPU", min_width=8, max_width=28),
                Column("mem", "MEM", min_width=7, max_width=13, align="right"),
                Column("state", "STATE", min_width=5, max_width=16),
                Column("features", "FEATURES", min_width=5, max_width=32),
            ],
        )

    return run_watch(args, render_once)


def command_partitions(args: argparse.Namespace) -> int:
    def render_once() -> None:
        rows = sinfo_partitions(args.partition)
        if args.json:
            print_json([asdict(row) for row in rows])
            return
        if not rows:
            print("No partition information found.")
            return
        color = should_color(args.no_color)
        table_rows = []
        for row in rows:
            available_color = "green" if row.available.lower().startswith("up") else "red"
            table_rows.append(
                {
                    "partition": colorize(row.partition, "cyan", color),
                    "available": colorize(row.available, available_color, color),
                    "limit": row.time_limit,
                    "nodes": row.nodes,
                    "state": colorize(row.state, color_for_state(row.state), color),
                    "gres": color_resource_value(row.gres, color),
                    "nodelist": row.nodelist,
                }
            )
        print_table(
            table_rows,
            [
                Column("partition", "PART", min_width=5, max_width=18),
                Column("available", "UP", min_width=2, max_width=4),
                Column("limit", "LIMIT", min_width=5, max_width=12),
                Column("nodes", "NODES", min_width=5, max_width=6, align="right"),
                Column("state", "STATE", min_width=5, max_width=12),
                Column("gres", "GRES", min_width=4, max_width=24),
                Column("nodelist", "NODELIST", min_width=8, max_width=44),
            ],
        )

    return run_watch(args, render_once)


def command_why(args: argparse.Namespace) -> int:
    def render_once() -> None:
        jobs = squeue_jobs(
            user=args.user,
            all_users=args.all,
            states=["PD", "PENDING"],
            partition=args.partition,
        )
        grouped: dict[str, list[Job]] = defaultdict(list)
        for job in jobs:
            grouped[job.location].append(job)
        data = [
            {
                "reason": reason,
                "count": len(items),
                "jobs": [item.job_id for item in items],
                "examples": [item.name for item in items[:3]],
                "starts": [item.start_time for item in items if item.start_time != "-"],
            }
            for reason, items in grouped.items()
        ]
        data.sort(key=lambda item: (-item["count"], item["reason"]))
        if args.json:
            print_json(data)
            return
        if not data:
            who = "all users" if args.all else args.user
            print(f"No pending jobs found for {who}.")
            return
        color = should_color(args.no_color)
        rows = []
        for item in data:
            starts = [start for start in item["starts"] if start and start != "N/A"]
            rows.append(
                {
                    "reason": colorize(item["reason"], "yellow", color),
                    "count": colorize(str(item["count"]), "bold", color),
                    "jobs": colorize(join_short(item["jobs"], 5), "cyan", color),
                    "examples": join_short(item["examples"], 3),
                    "start": colorize(format_start_time(min(starts)) if starts else "-", "gray", color),
                }
            )
        print_table(
            rows,
            [
                Column("reason", "REASON", min_width=8, max_width=26),
                Column("count", "COUNT", min_width=5, max_width=6, align="right"),
                Column("jobs", "JOBS", min_width=6, max_width=32),
                Column("examples", "EXAMPLES", min_width=8, max_width=32),
                Column("start", "FIRST START", min_width=8, max_width=11),
            ],
        )

    return run_watch(args, render_once)


def add_why_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=current_user(), help="Query this user. Defaults to the current user.")
    parser.add_argument("-a", "--all", action="store_true", help="Show pending reasons for all users.")
    parser.add_argument("-p", "--partition", help="Only show jobs in this partition.")
    add_common_output_flags(parser)


def history_to_row(job: HistoryJob, *, color: bool) -> dict[str, str]:
    exit_color = "green" if job.exit_code in {"0:0", "0"} else "red"
    return {
        "job_id": colorize(job.job_id, "bold", color),
        "name": job.name,
        "partition": colorize(job.partition, "cyan", color),
        "state": colorize(job.state, color_for_state(job.state), color),
        "elapsed": job.elapsed,
        "cpus": color_resource_value(job.cpus, color),
        "tres": color_resource_value(job.alloc_tres, color),
        "exit": colorize(job.exit_code, exit_color, color),
    }


def command_history(args: argparse.Namespace) -> int:
    def render_once() -> None:
        rows = sacct_history(user=args.user, days=args.days)
        if args.json:
            print_json([asdict(row) for row in rows])
            return
        if not rows:
            print("No recent sacct records found.")
            return
        color = should_color(args.no_color)
        table_rows = [history_to_row(row, color=color) for row in rows]
        print_table(
            table_rows,
            [
                Column("job_id", "JOBID", min_width=6, max_width=14),
                Column("name", "NAME", min_width=8, max_width=24),
                Column("partition", "PART", min_width=5, max_width=12),
                Column("state", "STATE", min_width=7, max_width=18),
                Column("elapsed", "ELAPSED", min_width=7, max_width=12, align="right"),
                Column("cpus", "CPU", min_width=3, max_width=5, align="right"),
                Column("tres", "ALLOC TRES", min_width=8, max_width=36),
                Column("exit", "EXIT", min_width=4, max_width=8),
            ],
        )

    return run_watch(args, render_once)


def add_history_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=current_user(), help="Query this user. Defaults to the current user.")
    parser.add_argument("--days", type=positive_int, default=7, help="Query the last N days. Defaults to 7.")
    add_common_output_flags(parser)


def build_parser(prog: str = "jslurm") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Small CLI helpers for Slurm.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    queue = subparsers.add_parser("queue", aliases=["q"], help="Show jobs for the current or selected user.")
    add_queue_args(queue)
    queue.set_defaults(func=command_queue)

    avail = subparsers.add_parser("avail", aliases=["a"], help="Show currently available GPU node resources.")
    add_node_filter_args(avail, include_avail_defaults=True)
    avail.set_defaults(func=command_avail)

    nodes = subparsers.add_parser("nodes", aliases=["n"], help="Show a node resource overview.")
    add_node_filter_args(nodes)
    nodes.set_defaults(func=command_nodes)

    partitions = subparsers.add_parser("part", aliases=["p"], help="Show a partition/sinfo overview.")
    partitions.add_argument("-p", "--partition", help="Only show this partition.")
    add_common_output_flags(partitions)
    partitions.set_defaults(func=command_partitions)

    why = subparsers.add_parser("why", aliases=["w"], help="Summarize pending job reasons.")
    add_why_args(why)
    why.set_defaults(func=command_why)

    history = subparsers.add_parser("hist", aliases=["h"], help="Show recent sacct history.")
    add_history_args(history)
    history.set_defaults(func=command_history)

    return parser


def run_with_errors(func: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    try:
        return func(args)
    except SlurmError as exc:
        print(f"jslurm: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_with_errors(args.func, args)


def main_queue(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jqueue", description="Show the Slurm job queue.")
    add_queue_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_queue, args)


def main_avail(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="javail", description="Show currently available GPU resources.")
    add_node_filter_args(parser, include_avail_defaults=True)
    args = parser.parse_args(argv)
    return run_with_errors(command_avail, args)


def main_nodes(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jnodes", description="Show a Slurm node resource overview.")
    add_node_filter_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_nodes, args)


def main_partitions(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jpart", description="Show a Slurm partition overview.")
    parser.add_argument("-p", "--partition", help="Only show this partition.")
    add_common_output_flags(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_partitions, args)


def main_why(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jwhy", description="Summarize Slurm pending job reasons.")
    add_why_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_why, args)


def main_history(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jhist", description="Show recent Slurm accounting history.")
    add_history_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_history, args)


if __name__ == "__main__":
    raise SystemExit(main())
